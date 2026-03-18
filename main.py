from __future__ import annotations

from datetime import date
from html import escape
import math
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Guarantee project root is on sys.path when running `streamlit run app/main.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.bot_service import responder
from services.conciliacao_service import carregar_movimentos
from utils.analise import (
    calcular_diferenca,
    classificar_tipo,
    gerar_sugestoes,
)
from utils.financeiro import formatar_brl, gerar_fluxo_diario, gerar_fluxo_mensal
from utils.filtros import filtrar_dataframe

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
ICON_PATH = ASSETS_DIR / "icon.png"
LOGO_PATH = ASSETS_DIR / "logo.png"
CSS_PATH = ASSETS_DIR / "styles.css"

TABLE_PAGE_SIZES = [25, 50, 100, 200, 500]
MAIN_COLUMNS = [
    "id_movimento_bancario",
    "nm_estabelecimento",
    "nm_fantasia_estab",
    "dt_transacao",
    "ds_banco",
    "tp_movimento",
    "ds_trasacao",
    "vl_liquido",
    "ds_origem_destino",
    "ds_status_conciliacao",
    "nm_usuario",
]
ENRICHED_COLUMNS = [
    "tipo",
    "valor_formatado",
    "status_visual",
    "sugestao",
    "valor_editado",
    "diferenca",
    "status_calculado",
]


def formatar_cnpj(valor):
    digitos = "".join(ch for ch in str(valor or "") if ch.isdigit())
    if len(digitos) != 14:
        return str(valor or "-")
    return (
        f"{digitos[0:2]}.{digitos[2:5]}.{digitos[5:8]}/"
        f"{digitos[8:12]}-{digitos[12:14]}"
    )


def carregar_css():
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _normalizar_colunas(df):
    resultado = df.copy()
    resultado.columns = [str(col).strip().lower() for col in resultado.columns]
    if "ds_trasacao" not in resultado.columns and "ds_transacao" in resultado.columns:
        resultado["ds_trasacao"] = resultado["ds_transacao"]
    return resultado


def _garantir_coluna_unidade(df):
    resultado = df.copy()

    def _limpar_unidade(serie):
        serie_txt = serie.astype(str).str.strip()
        serie_txt = serie_txt.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA, "NaN": pd.NA})
        return serie_txt.fillna("SEM_UNIDADE")

    if "ds_unidade" in resultado.columns:
        resultado["ds_unidade"] = _limpar_unidade(resultado["ds_unidade"])
        return resultado

    coluna_nome = "nm_fantasia_estab" if "nm_fantasia_estab" in resultado.columns else None
    coluna_estab = "cd_estabelecimento" if "cd_estabelecimento" in resultado.columns else None
    coluna_empresa = "cd_empresa" if "cd_empresa" in resultado.columns else None

    if coluna_nome and coluna_estab:
        estab = resultado[coluna_estab].astype(str).str.strip()
        nome = resultado[coluna_nome].astype(str).str.strip()
        resultado["ds_unidade"] = (estab + " - " + nome).str.strip(" -")
    elif coluna_nome:
        resultado["ds_unidade"] = resultado[coluna_nome].astype(str).str.strip()
    elif coluna_estab:
        resultado["ds_unidade"] = resultado[coluna_estab].astype(str).str.strip()
    elif coluna_empresa:
        resultado["ds_unidade"] = resultado[coluna_empresa].astype(str).str.strip()
    else:
        resultado["ds_unidade"] = "SEM_UNIDADE"

    resultado["ds_unidade"] = _limpar_unidade(resultado["ds_unidade"])
    return resultado


@st.cache_data(show_spinner=False)
def carregar_dados_base():
    df = carregar_movimentos()
    if df is None:
        return pd.DataFrame()

    df = _normalizar_colunas(df)
    df = _garantir_coluna_unidade(df)

    if "dt_transacao" in df.columns:
        df["dt_transacao"] = pd.to_datetime(df["dt_transacao"], errors="coerce")

    if "vl_liquido" in df.columns:
        df["vl_liquido"] = pd.to_numeric(df["vl_liquido"], errors="coerce")

    return df


def _serie_com_padrao(df, coluna, padrao=""):
    if coluna in df.columns:
        return df[coluna]
    return pd.Series([padrao] * len(df), index=df.index)


def _texto_limpo(valor):
    if valor is None:
        return "-"
    try:
        if pd.isna(valor):
            return "-"
    except TypeError:
        pass
    except ValueError:
        pass
    except Exception:
        pass

    if isinstance(valor, float) and pd.isna(valor):
        return "-"
    texto = str(valor).strip()
    return texto if texto else "-"


def _celula_texto(valor, max_chars=64):
    texto = _texto_limpo(valor)
    curto = texto if len(texto) <= max_chars else f"{texto[: max_chars - 3]}..."
    return f"<span class='cell-text' title='{escape(texto)}'>{escape(curto)}</span>"


def _celula_data(valor):
    data = pd.to_datetime(valor, errors="coerce")
    if pd.isna(data):
        return "-"
    return data.strftime("%d/%m/%Y")


def _valor_numero(valor):
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(numero):
        return 0.0
    return numero


def _badge_tipo(valor):
    numero = _valor_numero(valor)
    if numero > 0:
        return "<span class='badge badge-entrada'>Entrada</span>"
    if numero < 0:
        return "<span class='badge badge-saida'>Saida</span>"
    return "<span class='badge badge-neutro'>Neutro</span>"


def _badge_status(valor):
    texto = _texto_limpo(valor).lower()
    if "concili" in texto or texto == "ok":
        return "<span class='badge badge-ok'>Conciliado</span>"
    return "<span class='badge badge-divergente'>Divergente</span>"


def _celula_moeda(valor, saida=False):
    numero = _valor_numero(valor)
    numero_formatar = -abs(numero) if saida else numero
    if numero_formatar > 0:
        classe = "valor valor-positivo"
    elif numero_formatar < 0:
        classe = "valor valor-negativo"
    else:
        classe = "valor valor-neutro"
    return f"<span class='{classe}'>{formatar_brl(numero_formatar)}</span>"


def _celula_diferenca(valor):
    numero = _valor_numero(valor)
    if abs(numero) < 0.005:
        return "<span class='badge badge-ok'>OK</span>"
    return f"<span class='valor valor-negativo'>{formatar_brl(numero)}</span>"


def _renderizar_tabela_html(df_exibicao, *, table_id, caption="", height=500, empty_message="Sem dados para exibir."):
    if df_exibicao.empty:
        st.info(empty_message)
        return

    tabela_html = df_exibicao.to_html(index=False, escape=False, classes="fin-table", table_id=table_id, border=0)
    caption_html = f"<div class='table-caption'>{escape(caption)}</div>" if caption else ""
    st.markdown(
        (
            "<div class='table-shell'>"
            f"{caption_html}"
            f"<div class='table-scroll' style='max-height:{int(height)}px'>"
            f"{tabela_html}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _controles_paginacao(total_linhas, key_prefix):
    tamanho_key = f"{key_prefix}_page_size"
    pagina_key = f"{key_prefix}_page"

    if tamanho_key not in st.session_state or st.session_state[tamanho_key] not in TABLE_PAGE_SIZES:
        st.session_state[tamanho_key] = 100

    tamanho_pagina = int(st.session_state[tamanho_key])
    total_paginas = max(1, math.ceil(max(1, total_linhas) / tamanho_pagina))

    if pagina_key not in st.session_state:
        st.session_state[pagina_key] = 1
    if st.session_state[pagina_key] > total_paginas:
        st.session_state[pagina_key] = total_paginas
    if st.session_state[pagina_key] < 1:
        st.session_state[pagina_key] = 1

    col1, col2, col3 = st.columns([1.2, 1, 2.8])
    with col1:
        st.selectbox("Linhas por pagina", TABLE_PAGE_SIZES, key=tamanho_key)

    tamanho_pagina = int(st.session_state[tamanho_key])
    total_paginas = max(1, math.ceil(max(1, total_linhas) / tamanho_pagina))
    if st.session_state[pagina_key] > total_paginas:
        st.session_state[pagina_key] = total_paginas

    with col2:
        st.number_input("Pagina", min_value=1, max_value=total_paginas, step=1, key=pagina_key)

    pagina = int(st.session_state[pagina_key])
    inicio = (pagina - 1) * tamanho_pagina
    fim = min(inicio + tamanho_pagina, total_linhas)
    primeiro = inicio + 1 if total_linhas else 0

    with col3:
        st.markdown(
            (
                "<div class='table-meta'>"
                f"Exibindo {primeiro} a {fim} de {total_linhas} registros filtrados"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    return inicio, fim


def exibir_tabela_transacoes_html(df_resultado):
    if df_resultado.empty:
        st.info("Sem transacoes no filtro atual.")
        return

    inicio, fim = _controles_paginacao(len(df_resultado), "tbl_transacoes")
    fatia = df_resultado.iloc[inicio:fim].copy()

    valor_base = pd.to_numeric(_serie_com_padrao(fatia, "vl_liquido", 0.0), errors="coerce").fillna(0.0)
    valor_editado = pd.to_numeric(_serie_com_padrao(fatia, "valor_editado", 0.0), errors="coerce").fillna(0.0)
    diferenca = pd.to_numeric(_serie_com_padrao(fatia, "diferenca", 0.0), errors="coerce").fillna(0.0)
    if "status_visual" in fatia.columns:
        status_ref = fatia["status_visual"]
    else:
        status_ref = _serie_com_padrao(fatia, "status_calculado", "Divergente")

    df_exibicao = pd.DataFrame(
        {
            "ID": _serie_com_padrao(fatia, "id_movimento_bancario").map(lambda x: _celula_texto(x, max_chars=28)),
            "EMPRESA": _serie_com_padrao(fatia, "nm_fantasia_estab").map(_celula_texto),
            "ESTABELECIMENTO": _serie_com_padrao(fatia, "nm_estabelecimento").map(_celula_texto),
            "DATA": _serie_com_padrao(fatia, "dt_transacao").map(_celula_data),
            "BANCO": _serie_com_padrao(fatia, "ds_banco").map(_celula_texto),
            "MOVIMENTO": _serie_com_padrao(fatia, "tp_movimento").map(_celula_texto),
            "TIPO": valor_editado.map(_badge_tipo),
            "TRANSACAO": _serie_com_padrao(fatia, "ds_trasacao").map(lambda x: _celula_texto(x, max_chars=72)),
            "ORIGEM/DESTINO": _serie_com_padrao(fatia, "ds_origem_destino").map(_celula_texto),
            "VALOR SISTEMA": valor_base.map(_celula_moeda),
            "VALOR EDITADO": valor_editado.map(_celula_moeda),
            "DIFERENCA": diferenca.map(_celula_diferenca),
            "STATUS": status_ref.map(_badge_status),
            "SUGESTAO": _serie_com_padrao(fatia, "sugestao").map(lambda x: _celula_texto(x, max_chars=82)),
            "USUARIO": _serie_com_padrao(fatia, "nm_usuario").map(_celula_texto),
        }
    )

    _renderizar_tabela_html(
        df_exibicao,
        table_id="tabela-transacoes-html",
        caption="Visual premium para analise rapida de entradas, saidas e divergencias.",
        height=520,
    )


def exibir_tabela_fluxo_html(df_fluxo, *, mensal=False, table_id="tabela-fluxo", height=400):
    if df_fluxo.empty:
        st.info("Sem dados para exibir no periodo selecionado.")
        return

    base = df_fluxo.copy()
    if mensal:
        referencia = _serie_com_padrao(base, "mes_referencia").map(_celula_texto)
        primeira_coluna = "MES REFERENCIA"
    else:
        referencia = _serie_com_padrao(base, "data").map(_celula_data)
        primeira_coluna = "DATA"

    saldo_inicial = pd.to_numeric(_serie_com_padrao(base, "saldo_inicial", 0.0), errors="coerce").fillna(0.0)
    entradas = pd.to_numeric(_serie_com_padrao(base, "entradas", 0.0), errors="coerce").fillna(0.0)
    saidas = pd.to_numeric(_serie_com_padrao(base, "saidas", 0.0), errors="coerce").fillna(0.0)
    saldo_final = pd.to_numeric(_serie_com_padrao(base, "saldo_final", 0.0), errors="coerce").fillna(0.0)

    df_exibicao = pd.DataFrame(
        {
            primeira_coluna: referencia,
            "SALDO INICIAL": saldo_inicial.map(_celula_moeda),
            "ENTRADAS": entradas.map(_celula_moeda),
            "SAIDAS": saidas.map(lambda v: _celula_moeda(v, saida=True)),
            "SALDO FINAL": saldo_final.map(_celula_moeda),
        }
    )

    _renderizar_tabela_html(
        df_exibicao,
        table_id=table_id,
        caption="Fluxo consolidado do periodo selecionado.",
        height=height,
    )


def renderizar_secao_html(titulo, subtitulo=""):
    subtitulo_html = f"<p>{escape(subtitulo)}</p>" if subtitulo else ""
    st.markdown(
        (
            "<div class='section-head'>"
            f"<h3>{escape(titulo)}</h3>"
            f"{subtitulo_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def exibir_cards_html(itens):
    if not itens:
        return

    cols = st.columns(len(itens))
    for col, item in zip(cols, itens):
        label = escape(str(item.get("label", "")))
        valor = escape(str(item.get("valor", "-")))
        detalhe = escape(str(item.get("detalhe", "")))
        tone = str(item.get("tone", "neutral")).strip().lower()
        classe = f"ux-card tone-{tone}" if tone else "ux-card"
        detalhe_html = f"<div class='ux-detail'>{detalhe}</div>" if detalhe else ""
        col.markdown(
            (
                f"<div class='{classe}'>"
                f"<div class='ux-label'>{label}</div>"
                f"<div class='ux-value'>{valor}</div>"
                f"{detalhe_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def renderizar_banner_status(diferenca):
    conciliado = abs(float(diferenca)) < 0.01
    classe = "status-banner ok" if conciliado else "status-banner divergente"
    texto = "Tudo conciliado com o extrato informado." if conciliado else "Diferenca encontrada na conciliacao."
    st.markdown(
        (
            f"<div class='{classe}'>"
            f"<span>{escape(texto)}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _mascara_conciliado(df):
    if df.empty:
        return pd.Series(dtype=bool, index=df.index)

    if "status_calculado" in df.columns:
        status_calc = df["status_calculado"].astype(str).str.strip().str.lower()
        return status_calc.eq("ok") | status_calc.str.contains("concili", na=False)

    if "ds_status_conciliacao" in df.columns:
        status_origem = df["ds_status_conciliacao"].astype(str).str.strip().str.lower()
        return status_origem.str.contains("conciliado", na=False)

    return pd.Series([False] * len(df), index=df.index)


def renderizar_check_conciliacao(df_resultado):
    renderizar_secao_html("Check de Conciliacao", "Veja ate qual data os lancamentos estao conciliados.")

    if df_resultado.empty or "dt_transacao" not in df_resultado.columns:
        st.info("Sem dados para verificar conciliacao por data.")
        return

    datas = pd.to_datetime(df_resultado["dt_transacao"], errors="coerce").dropna()
    if datas.empty:
        st.info("Nao ha datas validas para checagem de conciliacao.")
        return

    data_min = datas.min().date()
    data_max = datas.max().date()
    chave_data = "check_data_conciliado_ate"

    valor_atual = st.session_state.get(chave_data)
    if valor_atual is None or valor_atual < data_min or valor_atual > data_max:
        st.session_state[chave_data] = data_max

    data_limite = st.date_input(
        "Conciliado ate qual data?",
        min_value=data_min,
        max_value=data_max,
        key=chave_data,
    )

    base = df_resultado.copy()
    base["dt_ref"] = pd.to_datetime(base["dt_transacao"], errors="coerce")
    base = base.dropna(subset=["dt_ref"])
    base = base[base["dt_ref"].dt.date <= data_limite].copy()

    if base.empty:
        st.info("Nao existem lancamentos ate a data selecionada.")
        return

    mascara_conc = _mascara_conciliado(base)
    total = int(len(base))
    qtd_conciliado = int(mascara_conc.sum())
    qtd_pendente = int(total - qtd_conciliado)
    cobertura = float((qtd_conciliado / total) * 100) if total else 0.0

    base["data_dia"] = base["dt_ref"].dt.date
    resumo_dia = pd.DataFrame({"data": base["data_dia"], "conciliado": mascara_conc.values}).groupby("data")[
        "conciliado"
    ].all()
    dias_totalmente_conciliados = resumo_dia[resumo_dia].index.tolist()
    ultima_data_totalmente_conciliada = max(dias_totalmente_conciliados) if dias_totalmente_conciliados else None
    ultima_data_txt = (
        pd.to_datetime(ultima_data_totalmente_conciliada).strftime("%d/%m/%Y")
        if ultima_data_totalmente_conciliada
        else "Nenhum dia 100% conciliado"
    )

    exibir_cards_html(
        [
            {"label": "Registros ate a data", "valor": f"{total}", "tone": "neutral"},
            {"label": "Conciliados", "valor": f"{qtd_conciliado}", "tone": "positive"},
            {"label": "Pendentes", "valor": f"{qtd_pendente}", "tone": "negative" if qtd_pendente else "positive"},
            {
                "label": "Cobertura",
                "valor": f"{cobertura:.1f}%",
                "detalhe": f"Ultimo dia 100%: {ultima_data_txt}",
                "tone": "primary",
            },
        ]
    )

    data_limite_txt = pd.to_datetime(data_limite).strftime("%d/%m/%Y")
    if qtd_pendente == 0:
        st.markdown(
            (
                "<div class='status-banner ok'>"
                f"Ate {data_limite_txt}, todos os lancamentos estao conciliados."
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            (
                "<div class='status-banner divergente'>"
                f"Ate {data_limite_txt}, existem {qtd_pendente} lancamentos pendentes de conciliacao."
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def obter_contas(df):
    if "ds_conta_bancaria" not in df.columns:
        return []

    contas = df["ds_conta_bancaria"].dropna().astype(str).str.strip()
    contas = contas[contas != ""]
    return sorted(contas.unique().tolist())


def obter_unidades(df):
    if "ds_unidade" not in df.columns:
        return []

    unidades = df["ds_unidade"].dropna().astype(str).str.strip()
    unidades = unidades[unidades != ""]
    return sorted(unidades.unique().tolist())


def limites_data(df):
    if "dt_transacao" not in df.columns or df["dt_transacao"].dropna().empty:
        hoje = date.today()
        return hoje, hoje

    serie = df["dt_transacao"].dropna()
    return serie.min().date(), serie.max().date()


def max_valor_abs(df):
    if "vl_liquido" not in df.columns:
        return 1000.0

    serie = pd.to_numeric(df["vl_liquido"], errors="coerce").abs().dropna()
    if serie.empty:
        return 1000.0

    return max(float(serie.max()), 1000.0)


def resetar_filtros(data_min, data_max, valor_max):
    st.session_state["flt_data_inicio"] = data_min
    st.session_state["flt_data_fim"] = data_max
    st.session_state["flt_valor_min"] = 0.0
    st.session_state["flt_valor_max"] = valor_max
    st.session_state["flt_texto"] = ""
    st.session_state["flt_tipo"] = "Todos"
    st.session_state["flt_status"] = "Todos"
    st.session_state["flt_busca_inteligente"] = ""


def montar_sidebar(df_base):
    unidades = obter_unidades(df_base)
    if not unidades:
        st.error("Nao foi possivel montar filtro de unidade. Coluna ds_unidade ausente.")
        st.stop()

    if st.session_state.get("flt_unidade") not in unidades:
        st.session_state["flt_unidade"] = unidades[0]

    with st.sidebar:
        st.markdown(
            (
                "<div class='sidebar-brand'>"
                "<div class='sidebar-title'>Conciliacao Inteligente</div>"
                "<div class='sidebar-subtitle'>Filtros avancados para analise rapida</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-group'>Unidade e conta</div>", unsafe_allow_html=True)
        unidade = st.selectbox("Unidade", unidades, key="flt_unidade")
        df_unidade = df_base[df_base["ds_unidade"].astype(str).str.strip() == str(unidade).strip()].copy()

        contas = obter_contas(df_unidade)
        if not contas:
            st.error("Nao existem contas para a unidade selecionada.")
            st.stop()

        if st.session_state.get("flt_conta") not in contas:
            st.session_state["flt_conta"] = contas[0]
        conta = st.selectbox("Conta", contas, key="flt_conta")

        df_conta = df_unidade[df_unidade["ds_conta_bancaria"].astype(str).str.strip() == str(conta).strip()].copy()
        data_min, data_max = limites_data(df_conta)
        valor_max = max_valor_abs(df_conta)

        if (
            st.session_state.get("flt_prev_unidade") != unidade
            or st.session_state.get("flt_prev_conta") != conta
        ):
            resetar_filtros(data_min, data_max, valor_max)
            st.session_state["flt_prev_unidade"] = unidade
            st.session_state["flt_prev_conta"] = conta

        if st.session_state.pop("_limpar_filtros_pending", False):
            resetar_filtros(data_min, data_max, valor_max)

        st.markdown("<div class='sidebar-group'>Periodo</div>", unsafe_allow_html=True)
        st.date_input("Data inicial", key="flt_data_inicio")
        st.date_input("Data final", key="flt_data_fim")
        st.markdown("<div class='sidebar-group'>Valores e classificacao</div>", unsafe_allow_html=True)
        st.number_input("Valor minimo", min_value=0.0, step=100.0, format="%.2f", key="flt_valor_min")
        st.number_input("Valor maximo", min_value=0.0, step=100.0, format="%.2f", key="flt_valor_max")
        st.selectbox("Tipo", ["Todos", "Entrada", "Saida"], key="flt_tipo")
        st.selectbox("Status", ["Todos", "Conciliado", "Divergente"], key="flt_status")
        st.markdown("<div class='sidebar-group'>Busca</div>", unsafe_allow_html=True)
        st.text_input("Busca", key="flt_texto", placeholder="Fornecedor ou descricao")
        st.text_input(
            "Busca inteligente",
            key="flt_busca_inteligente",
            placeholder="pix 200 | saida fornecedor x | entrada 1000",
        )

        if st.button("Limpar filtros", use_container_width=True):
            st.session_state["_limpar_filtros_pending"] = True
            st.rerun()

    if st.session_state["flt_data_inicio"] > st.session_state["flt_data_fim"]:
        st.error("Data inicial deve ser menor ou igual a data final.")
        st.stop()

    if (
        st.session_state["flt_valor_max"] > 0
        and st.session_state["flt_valor_min"] > st.session_state["flt_valor_max"]
    ):
        st.error("Valor minimo nao pode ser maior que valor maximo.")
        st.stop()

    filtros = {
        "unidade": st.session_state["flt_unidade"],
        "conta": st.session_state["flt_conta"],
        "data_inicio": st.session_state["flt_data_inicio"],
        "data_fim": st.session_state["flt_data_fim"],
        "valor_min": st.session_state["flt_valor_min"],
        "valor_max": st.session_state["flt_valor_max"],
        "texto": st.session_state["flt_texto"],
        "tipo": st.session_state["flt_tipo"],
        "status": st.session_state["flt_status"],
        "busca_inteligente": st.session_state["flt_busca_inteligente"],
    }

    return filtros


def preparar_tabela(df_filtrado):
    base = _normalizar_colunas(df_filtrado)

    for coluna in MAIN_COLUMNS:
        if coluna not in base.columns:
            base[coluna] = pd.NA

    base = base[MAIN_COLUMNS].copy()
    base = classificar_tipo(base)
    base = calcular_diferenca(base)
    base = gerar_sugestoes(base)
    return base


def assinatura_editor(df):
    if df.empty or "id_movimento_bancario" not in df.columns:
        return "empty"

    ids = df["id_movimento_bancario"].astype(str)
    return str(pd.util.hash_pandas_object(ids, index=False).sum())


def aplicar_edicao(df_base_analise, df_editor):
    resultado = df_base_analise.copy()

    if df_editor.empty:
        return resultado

    if "id_movimento_bancario" in df_editor.columns and "id_movimento_bancario" in resultado.columns:
        serie_editada = pd.to_numeric(df_editor["valor_editado"], errors="coerce")
        mapa = pd.Series(serie_editada.values, index=df_editor["id_movimento_bancario"].values)
        resultado["valor_editado"] = resultado["id_movimento_bancario"].map(mapa).fillna(resultado["valor_editado"])
    else:
        resultado["valor_editado"] = pd.to_numeric(df_editor.get("valor_editado"), errors="coerce").fillna(
            resultado["valor_editado"]
        )

    resultado = calcular_diferenca(resultado)
    resultado = gerar_sugestoes(resultado)
    return resultado


def exibir_topo(df_conta, unidade, conta, total_contas_unidade):
    registro = df_conta.iloc[0] if not df_conta.empty else pd.Series(dtype=object)
    cnpj = registro.get("cnpj_limpo") or registro.get("cnpj") or "-"
    banco = registro.get("ds_banco", "-")
    fantasia = registro.get("nm_fantasia_estab", "-")
    renderizar_secao_html("Dados da Conta", "Contexto da unidade e conta selecionadas para conciliacao.")
    exibir_cards_html(
        [
            {"label": "Unidade", "valor": str(unidade), "tone": "primary"},
            {"label": "Conta", "valor": str(conta), "detalhe": f"{int(total_contas_unidade)} contas na unidade", "tone": "neutral"},
            {"label": "CNPJ", "valor": formatar_cnpj(cnpj), "tone": "neutral"},
            {"label": "Banco", "valor": str(banco), "detalhe": str(fantasia), "tone": "neutral"},
        ]
    )


def exibir_cards_trimestre(df_conta):
    if "dt_transacao" not in df_conta.columns or df_conta["dt_transacao"].dropna().empty:
        ano_referencia = date.today().year
    else:
        ano_referencia = int(pd.to_datetime(df_conta["dt_transacao"], errors="coerce").dropna().dt.year.max())

    renderizar_secao_html(
        f"Janeiro, Fevereiro e Marco ({ano_referencia})",
        "Resumo mensal com entradas, saidas e saldos.",
    )

    base = preparar_tabela(df_conta)
    diario = gerar_fluxo_diario(base, saldo_inicial_base=0.0)

    if diario.empty:
        st.info("Sem dados para resumo de trimestre.")
        return

    diario["data"] = pd.to_datetime(diario["data"], errors="coerce")

    meses = [(1, "Janeiro"), (2, "Fevereiro"), (3, "Marco")]
    cols = st.columns(3)

    for col, (mes, nome) in zip(cols, meses):
        bloco = diario[diario["data"].dt.month == mes]

        if bloco.empty:
            entradas = saidas = saldo_inicial = saldo_final = 0.0
        else:
            entradas = float(bloco["entradas"].sum())
            saidas = float(bloco["saidas"].sum())
            saldo_inicial = float(bloco["saldo_inicial"].iloc[0])
            saldo_final = float(bloco["saldo_final"].iloc[-1])

        col.markdown(
            (
                "<div class='card'>"
                f"<div style='font-weight:700;margin-bottom:8px'>{nome}</div>"
                f"<div class='positivo'>Entradas: {formatar_brl(entradas)}</div>"
                f"<div class='negativo'>Saidas: {formatar_brl(saidas)}</div>"
                f"<div>Saldo inicial: <b>{formatar_brl(saldo_inicial)}</b></div>"
                f"<div>Saldo final: <b>{formatar_brl(saldo_final)}</b></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


page_config = {
    "page_title": "Conciliacao Inteligente",
    "layout": "wide",
}
if ICON_PATH.exists():
    page_config["page_icon"] = str(ICON_PATH)

st.set_page_config(**page_config)
carregar_css()

st.markdown("<div class='app-header'>", unsafe_allow_html=True)
col_logo, col_titulo = st.columns([1.2, 6.8])
with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=210)
with col_titulo:
    st.title("Conciliacao Inteligente")
    st.caption("Dashboard financeiro com filtros avancados, tabela premium e conferencia de extrato.")
st.markdown("</div>", unsafe_allow_html=True)

try:
    df_base = carregar_dados_base()
except Exception as exc:
    st.error("Erro ao carregar movimentos do banco.")
    st.exception(exc)
    st.stop()

if df_base.empty:
    st.warning("Nenhum movimento retornado pela consulta.")
    st.stop()

filtros = montar_sidebar(df_base)
unidade = str(filtros["unidade"]).strip()
conta = str(filtros["conta"]).strip()

df_unidade = df_base[df_base["ds_unidade"].astype(str).str.strip() == unidade].copy()
df_conta = df_unidade[df_unidade["ds_conta_bancaria"].astype(str).str.strip() == conta].copy()
df_filtrado = filtrar_dataframe(df_base, filtros)
df_analise_base = preparar_tabela(df_filtrado)

editor_columns = MAIN_COLUMNS + ENRICHED_COLUMNS
assinatura_atual = assinatura_editor(df_analise_base)
if (
    st.session_state.get("editor_assinatura") != assinatura_atual
    or "editor_transacoes_df" not in st.session_state
):
    st.session_state["editor_assinatura"] = assinatura_atual
    st.session_state["editor_transacoes_df"] = df_analise_base[editor_columns].copy()

# Defaults for manual statement section (used in the flow tables and editor calculations).
st.session_state.setdefault("manual_saldo_inicial", 0.0)
st.session_state.setdefault("manual_entradas", 0.0)
st.session_state.setdefault("manual_saidas", 0.0)

exibir_topo(df_conta, unidade, conta, total_contas_unidade=len(obter_contas(df_unidade)))

renderizar_secao_html(
    "Tabela Principal de Transacoes",
    f"Unidade: {unidade} | Conta: {conta} | Exibindo {len(df_analise_base)} de {len(df_base)} registros.",
)

tab_visual, tab_editor = st.tabs(["Visao Premium HTML", "Editor de Extrato"])

editor_view_columns = [
    "id_movimento_bancario",
    "nm_fantasia_estab",
    "nm_estabelecimento",
    "dt_transacao",
    "ds_trasacao",
    "tp_movimento",
    "vl_liquido",
    "valor_editado",
    "diferenca",
    "status_visual",
    "status_calculado",
]

with tab_editor:
    st.caption("Edite apenas o campo VALOR EDITADO para simular ajuste de extrato.")
    df_editor = st.data_editor(
        st.session_state["editor_transacoes_df"][editor_view_columns],
        key="editor_transacoes",
        height=500,
        width="stretch",
        disabled=[col for col in editor_view_columns if col != "valor_editado"],
        column_config={
            "id_movimento_bancario": st.column_config.TextColumn("ID"),
            "nm_fantasia_estab": st.column_config.TextColumn("EMPRESA"),
            "nm_estabelecimento": st.column_config.TextColumn("ESTABELECIMENTO"),
            "dt_transacao": st.column_config.DatetimeColumn("DATA", format="DD/MM/YYYY"),
            "ds_trasacao": st.column_config.TextColumn("TRANSACAO"),
            "tp_movimento": st.column_config.TextColumn("MOVIMENTO"),
            "vl_liquido": st.column_config.NumberColumn("VALOR SISTEMA", format="%.2f"),
            "valor_editado": st.column_config.NumberColumn("VALOR EDITADO", format="%.2f", step=0.01),
            "diferenca": st.column_config.NumberColumn("DIFERENCA", format="%.2f"),
            "status_visual": st.column_config.TextColumn("STATUS BASE"),
            "status_calculado": st.column_config.TextColumn("STATUS"),
        },
    )

df_resultado = aplicar_edicao(df_analise_base, df_editor[["id_movimento_bancario", "valor_editado"]])
st.session_state["editor_transacoes_df"] = df_resultado[editor_columns].copy()

with tab_visual:
    exibir_tabela_transacoes_html(df_resultado)

entradas_total = (
    float(df_resultado.loc[df_resultado["valor_editado"] > 0, "valor_editado"].sum()) if not df_resultado.empty else 0.0
)
saidas_total = (
    float(abs(df_resultado.loc[df_resultado["valor_editado"] < 0, "valor_editado"].sum())) if not df_resultado.empty else 0.0
)
saldo_sistema = float(df_resultado.get("valor_editado", pd.Series(dtype=float)).sum()) if not df_resultado.empty else 0.0
qtd_divergencias = int((df_resultado.get("status_calculado", pd.Series(dtype=str)) != "OK").sum()) if not df_resultado.empty else 0

renderizar_secao_html("Indicadores", "Resumo dinamico apos filtros e ajustes manuais.")
exibir_cards_html(
    [
        {"label": "Entradas", "valor": formatar_brl(entradas_total), "tone": "positive"},
        {"label": "Saidas", "valor": formatar_brl(saidas_total), "tone": "negative"},
        {"label": "Saldo Sistema", "valor": formatar_brl(saldo_sistema), "tone": "primary"},
        {"label": "Divergencias", "valor": str(qtd_divergencias), "tone": "warning"},
    ]
)
renderizar_check_conciliacao(df_resultado)

saldo_inicial_base = float(st.session_state.get("manual_saldo_inicial", 0.0))
df_diario = gerar_fluxo_diario(df_resultado, saldo_inicial_base=saldo_inicial_base)
df_mensal = gerar_fluxo_mensal(df_resultado, saldo_inicial_base=saldo_inicial_base)

renderizar_secao_html("Fluxo Diario", "Movimentacao consolidada por dia.")
if df_diario.empty:
    st.info("Sem dados para fluxo diario no filtro atual.")
else:
    exibir_tabela_fluxo_html(
        df_diario,
        mensal=False,
        table_id="tabela-fluxo-diario",
        height=400,
    )

renderizar_secao_html("Fluxo Mensal", "Consolidado mensal no padrao de controladoria.")
if df_mensal.empty:
    st.info("Sem dados para fluxo mensal no filtro atual.")
else:
    exibir_tabela_fluxo_html(
        df_mensal,
        mensal=True,
        table_id="tabela-fluxo-mensal",
        height=380,
    )

exibir_cards_trimestre(df_conta)

with st.expander("Detalhar mes"):
    if df_mensal.empty:
        st.info("Sem dados de mes para detalhamento.")
    else:
        opcoes_mes = df_mensal["mes_referencia"].tolist()
        mes_selecionado = st.selectbox("Selecione o mes", opcoes_mes, key="mes_detalhe")

        periodo = pd.to_datetime(mes_selecionado, format="%d/%m/%Y", errors="coerce")
        if pd.isna(periodo):
            st.info("Mes invalido para detalhamento.")
        else:
            periodo = periodo.to_period("M")
            detalhe = df_diario.copy()
            detalhe["data"] = pd.to_datetime(detalhe["data"], errors="coerce")
            detalhe = detalhe[detalhe["data"].dt.to_period("M") == periodo].copy()

            if detalhe.empty:
                st.info("Sem dias para o mes selecionado.")
            else:
                exibir_tabela_fluxo_html(
                    detalhe,
                    mensal=False,
                    table_id="tabela-fluxo-detalhe-mes",
                    height=280,
                )

renderizar_secao_html("Conferencia de Extrato", "Informe o extrato manual para comparacao com o sistema.")
e1, e2, e3 = st.columns(3)
with e1:
    saldo_inicial = st.number_input("Saldo inicial", key="manual_saldo_inicial", step=100.0, format="%.2f")
with e2:
    entradas_extrato = st.number_input("Entradas extrato", key="manual_entradas", step=100.0, format="%.2f")
with e3:
    saidas_extrato = st.number_input("Saidas extrato", key="manual_saidas", step=100.0, format="%.2f")

saldo_final_extrato = float(saldo_inicial + entradas_extrato - saidas_extrato)
diferenca_extrato = float(saldo_final_extrato - saldo_sistema)
tom_diferenca = "positive" if abs(diferenca_extrato) < 0.01 else "negative"
exibir_cards_html(
    [
        {"label": "Saldo Sistema", "valor": formatar_brl(saldo_sistema), "tone": "primary"},
        {"label": "Saldo Extrato", "valor": formatar_brl(saldo_final_extrato), "tone": "neutral"},
        {"label": "Diferenca", "valor": formatar_brl(diferenca_extrato), "tone": tom_diferenca},
    ]
)
renderizar_banner_status(diferenca_extrato)

renderizar_secao_html("Assistente", "Faca perguntas sobre os registros filtrados.")
pergunta = st.text_input("Pergunta sobre os dados filtrados")

if pergunta and pergunta.strip():
    if df_resultado.empty:
        st.info("Nao ha registros para o assistente analisar.")
    else:
        with st.spinner("Consultando assistente..."):
            resposta = responder(pergunta.strip(), df_resultado)
        resposta_html = escape(str(resposta)).replace("\n", "<br>")
        st.markdown(f"<div class='assistant-answer'>{resposta_html}</div>", unsafe_allow_html=True)


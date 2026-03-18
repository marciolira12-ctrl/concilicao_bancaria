from openai import OpenAI
from settings import OPENAI_API_KEY, OPENAI_MODEL

SENSITIVE_COLUMNS = {
    "cnpj",
    "cnpj_limpo",
    "cd_cgc_movimento",
    "cd_pessoa_fisica",
    "cd_paciente",
    "nr_documento",
}

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def _montar_contexto(df):
    df_limpo = df.drop(columns=[col for col in SENSITIVE_COLUMNS if col in df.columns])
    return df_limpo.head(50).to_string(index=False)


def responder(pergunta, df):
    if client is None:
        return "OPENAI_API_KEY nao configurada."

    if df.empty:
        return "Nao ha movimentos para analisar."

    contexto = _montar_contexto(df)

    try:
        resposta = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Voce e especialista em conciliacao bancaria. "
                        "Responda de forma objetiva e use apenas o contexto fornecido."
                    ),
                },
                {"role": "user", "content": f"{contexto}\n\nPergunta: {pergunta}"},
            ],
            temperature=0.2,
        )
        return resposta.choices[0].message.content or "Nao consegui gerar resposta."
    except Exception as exc:
        return f"Falha ao consultar o assistente: {exc}"

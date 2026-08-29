"""
validacao.py

Regras de validação dos dados de transações do pipeline NeoPIX.
"""

import pandas as pd

from logging_config import get_logger


logger = get_logger("validacao")


STATUS_VALIDOS = [
    "completed",
    "failed",
    "pending",
    "reversed",
]


# Definir estes valores depois da exploração do dataset.
LIMITE_AMOUNT = 10050.0
LIMITE_PROCESSING_TIME_MS = 5000.0


def validar_status_permitido(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica registros cujo status não pertence ao domínio permitido.

    Args:
        df: DataFrame contendo as transações.

    Returns:
        DataFrame contendo apenas os registros com status inválido.
    """

    rejeitados = df[
        ~df["status"].isin(STATUS_VALIDOS)
    ].copy()

    for _, row in rejeitados.iterrows():
        logger.warning(
            "Registro rejeitado",
            extra={
                "transaction_id": row["transaction_id"],
                "motivo": "status_invalido",
            },
        )

    return rejeitados


def validar_amount_positivo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica registros com amount menor ou igual a zero.

    Args:
        df: DataFrame contendo as transações.

    Returns:
        DataFrame contendo registros com amount inválido.
    """

    rejeitados = df[
        df["amount"] <= 0
    ].copy()

    for _, row in rejeitados.iterrows():
        logger.warning(
            "Registro rejeitado",
            extra={
                "transaction_id": row["transaction_id"],
                "motivo": "amount_invalido",
            },
        )

    return rejeitados


def validar_amount_muito_alto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica registros cujo amount ultrapassa o limite definido.

    Args:
        df: DataFrame contendo as transações.

    Returns:
        DataFrame contendo registros com amount acima do limite.
    """

    rejeitados = df[
        df["amount"] > LIMITE_AMOUNT
    ].copy()

    for _, row in rejeitados.iterrows():
        logger.warning(
            "Registro rejeitado",
            extra={
                "transaction_id": row["transaction_id"],
                "motivo": "amount_muito_alto",
            },
        )

    return rejeitados


def validar_processing_time_positivo(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identifica registros com tempo de processamento negativo.

    Args:
        df: DataFrame contendo as transações.

    Returns:
        DataFrame contendo registros com tempo negativo.
    """

    rejeitados = df[
        df["processing_time_ms"] < 0
    ].copy()

    for _, row in rejeitados.iterrows():
        logger.warning(
            "Registro rejeitado",
            extra={
                "transaction_id": row["transaction_id"],
                "motivo": "processing_time_negativo",
            },
        )

    return rejeitados


def validar_processing_time_muito_alto(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identifica registros cujo tempo de processamento
    ultrapassa o limite definido.

    Args:
        df: DataFrame contendo as transações.

    Returns:
        DataFrame contendo registros com tempo de processamento
        acima do limite.
    """

    rejeitados = df[
        df["processing_time_ms"] > LIMITE_PROCESSING_TIME_MS
    ].copy()

    for _, row in rejeitados.iterrows():
        logger.warning(
            "Registro rejeitado",
            extra={
                "transaction_id": row["transaction_id"],
                "motivo": "processing_time_muito_alto",
            },
        )

    return rejeitados


def validar_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica registros cujo timestamp não pôde ser convertido
    e apresenta valor NaT.

    Args:
        df: DataFrame contendo as transações.

    Returns:
        DataFrame contendo registros com timestamp inválido.
    """

    rejeitados = df[
        df["timestamp"].isna()
    ].copy()

    for _, row in rejeitados.iterrows():
        logger.warning(
            "Registro rejeitado",
            extra={
                "transaction_id": row["transaction_id"],
                "motivo": "timestamp_invalido",
            },
        )

    return rejeitados


def validar_transaction_id_duplicado(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identifica registros com transaction_id duplicado.

    Args:
        df: DataFrame contendo as transações.

    Returns:
        DataFrame contendo todas as ocorrências de IDs duplicados.
    """

    rejeitados = df[
        df.duplicated(
            subset="transaction_id",
            keep=False,
        )
    ].copy()

    for _, row in rejeitados.iterrows():
        logger.warning(
            "Registro rejeitado",
            extra={
                "transaction_id": row["transaction_id"],
                "motivo": "transaction_id_duplicado",
            },
        )

    return rejeitados


def executar_validacoes(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa todas as validações de qualidade dos dados.

    Os registros que falharem em pelo menos uma regra são
    classificados como rejeitados.

    Args:
        df: DataFrame contendo as transações.

    Returns:
        Tupla contendo:

        - dados_validos: registros aprovados em todas as validações.
        - dados_rejeitados: registros rejeitados com a coluna
          motivo_rejeicao.
    """

    validacoes = [
        (
            validar_status_permitido(df),
            "status_invalido",
        ),
        (
            validar_amount_positivo(df),
            "amount_invalido",
        ),
        (
            validar_amount_muito_alto(df),
            "amount_muito_alto",
        ),
        (
            validar_processing_time_positivo(df),
            "processing_time_negativo",
        ),
        (
            validar_processing_time_muito_alto(df),
            "processing_time_muito_alto",
        ),
        (
            validar_timestamp(df),
            "timestamp_invalido",
        ),
        (
            validar_transaction_id_duplicado(df),
            "transaction_id_duplicado",
        ),
    ]

    rejeitados_list = []

    for registros, motivo in validacoes:

        if not registros.empty:
            registros = registros.copy()
            registros["motivo_rejeicao"] = motivo
            rejeitados_list.append(registros)

    if rejeitados_list:
        dados_rejeitados = pd.concat(
            rejeitados_list,
            ignore_index=True,
        )
    else:
        dados_rejeitados = pd.DataFrame(
            columns=[
                *df.columns,
                "motivo_rejeicao",
            ]
        )

    ids_rejeitados = set(
        dados_rejeitados["transaction_id"]
    )

    dados_validos = df[
        ~df["transaction_id"].isin(ids_rejeitados)
    ].copy()

    logger.info(
        "Validação concluída",
        extra={
            "total_registros": len(df),
            "registros_validos": len(dados_validos),
            "registros_rejeitados": len(dados_rejeitados),
        },
    )

    return dados_validos, dados_rejeitados
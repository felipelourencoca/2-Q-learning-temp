"""Configuração do pytest — opções customizadas."""


def pytest_addoption(parser):
    parser.addoption(
        "--max-steps",
        action="store",
        default=50,
        type=int,
        help="Máximo de passos por episódio (padrão: 50)",
    )
    parser.addoption(
        "--alpha",
        action="store",
        default=0.1,
        type=float,
        help="Taxa de aprendizado (padrão: 0.1)",
    )
    parser.addoption(
        "--gamma",
        action="store",
        default=0.9,
        type=float,
        help="Fator de desconto (padrão: 0.9)",
    )
    parser.addoption(
        "--epsilon",
        action="store",
        default=1.0,
        type=float,
        help="Taxa de exploração inicial (padrão: 1.0)",
    )
    parser.addoption(
        "--epsilon-decay",
        action="store",
        default=0.995,
        type=float,
        help="Fator de decaimento do epsilon (padrão: 0.995)",
    )
    parser.addoption(
        "--epsilon-min",
        action="store",
        default=0.01,
        type=float,
        help="Valor mínimo do epsilon (padrão: 0.01)",
    )

# Importação dos agentes
# ==========================
# Importa as classes dos agentes responsáveis por diferentes funções (ajuda, otimização, análise de dados, e gerenciamento).
# Importa também a classe ChatOpenAI para interação com o modelo GPT-4o.

from agents.helper_agent import HelperAgent
from agents.optimizer_agent import OptimizerAgent
from agents.data_analytics import DataAnalyticsAgent
from agents.manager_agent import ManagerAgent
from langchain_openai import ChatOpenAI  # Permite uso do GPT-4o e GPT-4o-mini


# ==========================
# Funções para criar agentes
# ==========================
# Cria uma instância do modelo GPT-4o (ou GPT-4o-mini) a ser utilizado pelos agentes.
model = ChatOpenAI(model="gpt-4o", temperature=1)

# Criação dos agentes
# ==========================
# Cria instâncias de cada agente (HelperAgent, OptimizerAgent, DataAnalyticsAgent) passando o modelo configurado.
helper_agent = HelperAgent(model)
optimizer_agent = OptimizerAgent(model)
data_analytics_agent = DataAnalyticsAgent(model)

# Definindo a equipe (team)
# ==========================
# Cria uma lista chamada 'team' contendo todos os agentes criados, prontos para serem gerenciados pelo supervisor.
team = [
    helper_agent.create_agent(),
    optimizer_agent.create_agent(),
    data_analytics_agent.create_agent(),
]

# Criando o supervisor (roteador)
# ==========================
# Cria uma instância do agente Manager (supervisor), passando o modelo e a equipe de agentes.
manager_agent = ManagerAgent(model, team)

# Criando o supervisor e compilando o workflow
# ==========================
# Cria o supervisor (roteador) usando o agente Manager, e então compila o workflow para que ele possa orquestrar as interações entre os agentes.
manager_supervisor = manager_agent.create_supervisor()
app = manager_supervisor.compile()

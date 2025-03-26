# Importação das bibliotecas necessárias
# ==========================
# Importa a função create_react_agent da biblioteca langgraph.prebuilt, que permite criar agentes reativos.

from langgraph.prebuilt import create_react_agent


# Definição da classe Agent
# ==========================
# A classe Agent é responsável por representar um agente com um modelo, nome, ferramentas e um prompt.
# Ela tem um método create_agent que cria um agente reativo usando os parâmetros fornecidos (modelo, nome, ferramentas e prompt).

class Agent:
    
    def __init__(self, model, name, tools, prompt):
        self.model = model  # Armazena o modelo a ser usado pelo agente
        self.name = name  # Armazena o nome do agente
        self.tools = tools  # Armazena as ferramentas que o agente pode usar (como funções externas)
        self.prompt = prompt  # Armazena o prompt que o agente usará para interagir com o modelo

    # Criação do agente
    # ==========================
    # Este método cria um agente reativo utilizando a função `create_react_agent` e os parâmetros passados para a classe Agent.
    def create_agent(self):
        return create_react_agent(
            self.model,  # Passa o modelo que será utilizado pelo agente
            name=self.name,  # Passa o nome do agente
            tools=self.tools,  # Passa as ferramentas que o agente pode usar
            prompt=self.prompt  # Passa o prompt que define o comportamento do agente
        )

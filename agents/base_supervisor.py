# Importação necessária para criação do supervisor
# ==========================
# Importa a função `create_supervisor` da biblioteca `langgraph_supervisor`, que cria o supervisor responsável por gerenciar os agentes.

from langgraph_supervisor import create_supervisor


# Definição da classe Supervisor
# ==========================
# A classe `Supervisor` representa um supervisor que irá gerenciar um time de agentes.
# Ela recebe três parâmetros no construtor: o modelo, a equipe (team) de agentes e o prompt que define o comportamento do supervisor.

class Supervisor:
    
    def __init__(self, model_name, team, prompt):
        # Atribui os parâmetros passados ao objeto `Supervisor`
        self.model = model_name,  # Atribui o nome do modelo ao supervisor
        self.team = team,  # Atribui a equipe de agentes ao supervisor
        self.prompt = prompt  # Atribui o prompt que será usado pelo supervisor
    
    # Função para criar o supervisor
    # ==========================
    # Este método chama a função `create_supervisor` passando a equipe de agentes, o modelo e o prompt.
    # A função `create_supervisor` vai criar o supervisor que gerenciará a equipe de agentes.

    def create_supervisor(self):
        return create_supervisor(
            self.team,  # Passa a equipe de agentes (team) para o supervisor
            model=self.model,  # Passa o modelo de linguagem que o supervisor usará
            prompt=self.prompt  # Passa o prompt que o supervisor usará para interagir com os agentes
        )

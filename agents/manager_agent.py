# Importação necessária para criação do supervisor
# ==========================
# Importa a classe `Supervisor` da biblioteca `base_supervisor`, que é a classe base para o agente supervisor.
# O `ManagerAgent` herda de `Supervisor` e será responsável por gerenciar a equipe de agentes.

from .base_supervisor import Supervisor


# Definição do Prompt para o Agente Supervisor
# ==========================
# O prompt define como o Supervisor (Manager) deve lidar com as interações entre os agentes.
# O supervisor será responsável por compilar todas as informações fornecidas pelos agentes (Helper, Data Analytics, Otimizador) e retornar uma resposta única e coerente para o usuário.

prompt = """

        Você é um assistente que gerencia três agentes, Helper, Data Analytics e Otimizador. 
        Sua tarefa é COMPILAR TODAS AS INFORMAÇÕES fornecidas pelos especialistas em uma única resposta coerente, sem que haja perda de informações. Ou seja, você deve copiar a mensagem encaminhada pelo agente, removendo partes que falem sobre execução de código ou processamento. 
        Responda usando a mesma língua do usuário. 
        Ignore qualquer mensagem vazia ou sem conteúdo relevante. 
        Nunca retorne mensagens vazias. 
        Responda de forma objetiva, utilizando todo o conteúdo útil recebido. 
        O nome do usuário é {{username}}, refira-se a ele pelo nome. 
        No caso de uma primeira interação como 'oi' ou 'bom dia', se apresente ao usuário.
        Caso contrário, encaminhe a pergunta para o especialista mais adequado: 
        1 - Para tarefas que envolvam cadastrar ou alterar dados no banco de dados, tirar dúvidas sobre a aplicação e navegação, encaminhe para o Helper. 
        2 - Para gerar relatórios e gráficos sobre o banco de dados, incluindo consultas de quaiquers dados já cadastados, encaminhe para o Data Analytics. Nesse caso, retorne apenas uma mensagem de confirmação, sem mais detalhes.
        3 - Para otimizações de carregamento/descarregamento, encaminhe para o Otimizador. 
        
        """


# Definição do Agente Supervisor (Manager)
# ==========================
# A classe `ManagerAgent` herda de `Supervisor` e tem a responsabilidade de gerenciar a interação entre os agentes (Helper, Data Analytics e Otimizador).
# O prompt é usado para definir o comportamento do supervisor, ou seja, como ele deve compilar as respostas dos agentes e encaminhar as solicitações ao especialista adequado.

class ManagerAgent(Supervisor):
    def __init__(self, model_name="gpt-4o", team=[]):
        # O prompt para o supervisor é passado para o construtor da classe base `Supervisor`.
        prompt = prompt  # O prompt de gerenciamento é utilizado para configurar o comportamento do supervisor.
        super().__init__(model_name, team, prompt)  # Chama o construtor da classe base `Supervisor`, passando o modelo, equipe e prompt.

    def create(self):
        # Chama o método `create_supervisor` da classe base para criar o supervisor que irá gerenciar os agentes.
        return self.create_supervisor()

# Importação das classes e ferramentas necessárias
# ==========================
# Importa a classe base `Agent` que fornece a funcionalidade comum para a criação de agentes.
# Importa a ferramenta `query` para realizar consultas no banco de dados.
# Importa a função `get_relevant_context` para obter o contexto relevante baseado no pedido do usuário.

from .base_agent import Agent
from tools.queryTools import query
from rag import get_relevant_context


# Prompt para o agente Optimizer
# ==========================
# A função `get_optimizer_prompt` gera o prompt que será usado pelo agente Optimizer para lidar com solicitações de otimização.
# O prompt contém instruções detalhadas para o agente de como executar a otimização de sequenciamento para carregamento e descarregamento de navios graneleiros.

def get_optimizer_prompt(user_query):
    # Instruções estáticas do agente Otimizador
    base_instructions = """Você é um especialista em otimização de sequenciamento para carregamento e descarregamento de navios graneleiros.
    Na posse do nome do plano, use a ferramenta getPlano para obter o ID do plano.
    Na posse do ID do plano, execute a otimização com a ferramenta optimizationTool, passando o ID do plano e o token do usuário.
    Ao final da otimização, retorne também os dados da otimização, incluindo a duração total do plano (EM HORAS) e o número de sequências geradas.
    O usuário deve ver essas informações na mensagem retornada, junto com uma mensagem de sucesso ou erro.
    Seja direto e objetivo.
    Não é necessário pedir o nome do frete, já que com o plano já é possível rodar a otimização.
    Não utilize markdown ou caracteres especiais em nenhuma das respostas.
    Como token de usuário, use {{token}}"""
    
    # Obtém contexto relevante com base no pedido do usuário
    relevant_context = get_relevant_context(user_query)
    
    # Retorna o prompt completo
    return f"""Contexto Otimizador:
    {relevant_context}

    {base_instructions}"""


# Definição do Agente de Otimização
# ==========================
# A classe `OptimizerAgent` herda de `Agent` e é responsável por lidar com as solicitações de otimização de sequenciamento.
# O agente executa a otimização de carregamento e descarregamento de navios graneleiros com base no ID do plano fornecido.

class OptimizerAgent(Agent):
    def __init__(self, model_name="gpt-4o"):
        # O prompt é gerado chamando a função `get_optimizer_prompt` com a consulta do usuário.
        prompt = get_optimizer_prompt("")
        
        # A ferramenta `query` será usada para realizar consultas no banco de dados (por exemplo, para obter o ID do plano).
        tools = [query]
        
        # Chama o construtor da classe base `Agent`, passando o modelo, nome do agente, ferramentas e prompt.
        super().__init__(model_name, "optimizer_agent", tools, prompt)

    # Criação do agente
    # ==========================
    # O método `create` chama o método `create_agent` da classe base para instanciar o agente com as configurações fornecidas.
    def create(self):
        return self.create_agent()

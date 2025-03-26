# Importação das classes e ferramentas necessárias
# ==========================
# Importa a classe base `Agent`, que provavelmente contém a lógica comum para criação de agentes.
# Importa a ferramenta `query` que será usada para realizar consultas no banco de dados.
# Importa a função `get_relevant_context` para obter o contexto relevante para a consulta do usuário.

from .base_agent import Agent
from tools.queryTools import query
from rag import get_relevant_context

# Prompt para o agente Data_Analytics
# ==========================
# A função `get_data_analytics_prompt` gera o prompt que será usado pelo agente DataAnalytics para lidar com consultas de dados.
# Esse prompt define como o agente deve gerar o código para diferentes cenários (gráfico ou consulta textual) e inclui regras sobre como acessar o banco de dados.

def get_data_analytics_prompt(user_query):

    base_instructions = """ATENÇÃO: É estritamente proibido utilizar qualquer dado fictício! Todas as informações devem ser obtidas diretamente do banco de dados por meio de uma query Django que filtre corretamente pelo usuário e pela empresa. Se a consulta não retornar dados, informe ao usuário que não há registros disponíveis, sem gerar dados simulados.
    Instruções Gerais:
    - Importe as models necessárias utilizando 'api.models'.
    - Garanta que todas as queries incluam os filtros por user_id e empresa_id.
    - Converta os dados para formatos numéricos apropriados, se necessário.
    - Execute o código usando a ferramenta de query e retorne o resultado da execução.
    - Em caso de erro na execução, informe o usuário detalhadamente.
    - Gere APENAS código, sem nenhum texto adicional ou explicações fora do bloco de código.
    Cenário 1 - Usuário solicita um gráfico:
    1) Gere um único bloco de código Python (delimitado por markdown) contendo:
    a) Uma query Django para obter os dados necessários.
    b) O trecho de código que utiliza Seaborn para gerar um gráfico completo e bem formatado.
    c) O código que converte o gráfico em bytes por meio de 'io.BytesIO' e o salvamento no model 'Imagem' (campo 'dados'); não utilize 'ContentFile'.
    2) Execute o código gerado usando a ferramenta de query.
    Cenário 2 - Usuário solicita uma consulta textual (por exemplo, para saber os planos existentes):
    1) Gere um único bloco de código Python (delimitado por markdown) contendo:
    a) Uma query Django para obter os dados necessários.
    b) O trecho de código que retorna uma string bem formatada com os resultados da consulta.
    c) O código que salva a string no campo no model 'Imagem' (campo 'texto').
    2) Execute o código gerado usando a ferramenta de query.
    Ao salvar texto ou imagem no banco, inclua o 'user_id'.
    Como user_id, use {{user}}, como empresa_id, use {{empresa}}.

    """
    
    # Obtém contexto relevante com base no pedido do usuário
    relevant_context = get_relevant_context(user_query)
    
    # Retorna o prompt completo, incluindo o contexto relevante e as instruções para o agente
    return f"""Contexto Data Analytics:
    {relevant_context}

    {base_instructions}"""


# Definição do Agente de Análise de Dados
# ==========================
# A classe `DataAnalyticsAgent` herda de `Agent` e é responsável por lidar com as consultas de dados e gerar os resultados adequados,
# como gráficos ou consultas textuais, dependendo do que o usuário solicitar.

class DataAnalyticsAgent(Agent):
    def __init__(self, model_name="gpt-4o"):
        # O prompt é gerado chamando a função `get_data_analytics_prompt` com a consulta do usuário.
        # O modelo de IA será o GPT-4o (ou outro modelo, se especificado).
        prompt = get_data_analytics_prompt("")
        
        # A ferramenta `query` será usada para realizar consultas no banco de dados.
        tools = [query]
        
        # Chama o construtor da classe base `Agent` passando o modelo, nome do agente, ferramentas e prompt.
        super().__init__(model_name, "data_analytics_agent", tools, prompt)

    # Criação do agente
    # ==========================
    # O método `create` chama o método `create_agent` da classe base `Agent` para criar uma instância do agente com as configurações definidas.
    def create(self):
        return self.create_agent()
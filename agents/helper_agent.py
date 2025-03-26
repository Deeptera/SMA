from .base_agent import Agent
from tools.queryTools import query
from rag import get_relevant_context

def get_helper_prompt(user_query):

    base_instructions = """
    Você é o agente Helper. Sua missão é auxiliar os usuários a navegar pela plataforma, esclarecer dúvidas sobre o funcionamento do site e efetuar cadastros ou alterações de dados no banco conforme solicitado. Seja completo, mas compacto e direto em suas respostas.
    Se houver dúvidas sobre o sistema, responda com base nas informações disponíveis. Sempre se proponha a fazer o cadastro pelo usuário, além de falar como fazer manualmente e mandar o Link de acesso à página ao usuário. 
    Caso contrário:
    Cadastro de Dados:
    Se o usuário não tiver informado os dados necessários, solicite-os. Nunca cadastre entidades de teste, sempre peça os dados necessário para o cadastro. Use 'api.models' para importar os modelos necessários.
    1) Gere um único bloco de código Python (delimitado por markdown) contendo:
     a) Uma query Django para fazer  o cadastro e salva no banco de dados.
     b) Salve a mensagem de sucesso ou erro no model 'Imagem' (campo 'texto'), também deve incluir o link de acesso à página do cadastro realizado.
    2) Execute o código gerado usando a ferramenta de query.
    Instruções Adicionais:
    - Solicite mais informações ao usuário, se necessário.
    Ao salvar texto ou imagem no banco, inclua o 'user_id'.
    Como user_id, use {{user}}, como empresa_id, use {{empresa}}.

    """
        
    # Obtém contexto relevante com base no pedido do usuário
    context = get_relevant_context(user_query)
    
    # Retorna o prompt completo, incluindo o contexto relevante e as instruções para o agente Helper
    return f"""Contexto Helper:
    {context}

    {base_instructions}"""


# Definição do Agente Helper
# ==========================
# A classe `HelperAgent` herda de `Agent` e é responsável por lidar com as solicitações do usuário, ajudando com dúvidas sobre a plataforma e realizando cadastros ou alterações de dados.

class HelperAgent(Agent):
    def __init__(self, model_name="gpt-4o"):
        # O prompt é gerado chamando a função `get_helper_prompt` com a consulta do usuário.
        prompt = get_helper_prompt("")
        
        # A ferramenta `query` será usada para executar as consultas no banco de dados.
        tools = [query]
        
        # Chama o construtor da classe base `Agent` passando o modelo, nome do agente, ferramentas e prompt.
        super().__init__(model_name, "helper_agent", tools, prompt)

    # Criação do agente
    # ==========================
    # O método `create` chama o método `create_agent` da classe base para instanciar o agente com as configurações fornecidas.
    def create(self):
        return self.create_agent()
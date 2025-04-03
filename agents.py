"""
Este arquivo contém a lógica para processar e extrair informações
da base de conhecimento, os prompts dos agentes utilizados no artigo
e a inicialização do workflow.

O código está organizado nas seguintes seções:
1. Importação de bibliotecas e frameworks
2. Definição dos diretórios e carregamento dos documentos
3. Criação do índice com FAISS para recuperação de contexto
4. Definição dos modelos de LLMs
5. Criação dos Prompts
6. Inicialização dos Agentes com os Prompts e Ferramentas
"""

# ====================================== #
# Importação de bibliotecas e frameworks #
# ====================================== #

# Importação dos módulos para comunicação com API das LLMs
from langchain_google_genai import ChatGoogleGenerativeAI # Permite uso do gemini
from langchain_anthropic import ChatAnthropic # Permite uso do Claude
from langchain_openai import ChatOpenAI # Permite uso do GPT-4o e GPT-4o-mini
# Importação dos módulos para criação dos agentes
from langgraph_supervisor import create_supervisor # Criação do supervisor
from langgraph.prebuilt import create_react_agent # Criação dos agentes
# Importação dos módulos para criação do embeddings e recuperação de contexto
from langchain.document_loaders import TextLoader # Carregamento de documentos
from langchain_openai import OpenAIEmbeddings # Embeddings para recuperação de contexto
from langchain_community.vectorstores import FAISS # Índice FAISS para recuperação de contexto
# Importação das ferramentas dos agentes, criadas pelos próprios desenvolvedores
from api.agentes.tools import optimizationTools, queryTools, getPlano # Ferramentas para os agentes
# Importação de módulos para manipulação de diretórios e variáveis de ambiente
import os # Biblioteca para manipulação de diretórios do sistema operacional
from dotenv import load_dotenv # Biblioteca para carregar variáveis de ambiente
load_dotenv() # Carrega variáveis de ambiente, como chaves API

# ====================================================== #
# Definição dos diretórios e carregamento dos documentos #
# ====================================================== #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

# Carrega o texto que contém informações sobre os modelos das entidades do banco
loader_banco = TextLoader(os.path.join(DOCS_DIR, "banco.txt"), encoding="utf-8")
docs_banco = loader_banco.load()

# Carrega o texto que contém informações sobre metodologia de geração de gráficos (Extreme Presentation)
loader_graficos = TextLoader(os.path.join(DOCS_DIR, "graficos.txt"), encoding="utf-8")
docs_graficos = loader_graficos.load()

# Carrega o texto que contém informações sobre uso do sistema
loader_manual = TextLoader(os.path.join(DOCS_DIR, "manual_do_usuario.txt"), encoding="utf-8")
docs_manual = loader_manual.load()

# Carrega o texto que contém informações sobre a documentação do sistema
loader_documentacao = TextLoader(os.path.join(DOCS_DIR, "documentacao.txt"), encoding="utf-8")
docs_documentacao = loader_documentacao.load()

# Carrega o texto que contém informações sobre as rotas da aplicação. Feature que não foi colocada no artigo, mas já existe no sistema (o agente direciona para a página relevante).
loader_links = TextLoader(os.path.join(DOCS_DIR, "links.txt"), encoding="utf-8")
docs_links = loader_links.load()

# Junta todos os documentos para indexação
docs = docs_banco + docs_graficos + docs_manual + docs_documentacao + docs_links

# ======================================================== #
# Criação do índice com FAISS para recuperação de contexto #
# ======================================================== #

embeddings = OpenAIEmbeddings()

# Cria ou carrega o índice FAISS com persistência local
if os.path.exists(INDEX_DIR):
    # Carrega o índice salvo localmente
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    print("Índice FAISS carregado do diretório local.")
else:
    # Cria o índice a partir dos documentos e o salva localmente
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print("Índice FAISS criado e salvo localmente.")

def get_relevant_context(query, k=8):
    """
    Função para recuperar o contexto relevante para uma query utilizando FAISS.
    Retorna um texto com os conteúdos dos k documentos mais relevantes.
    """
    relevant_docs = vectorstore.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    return context

# ============================= #
# Definição dos modelos de LLMs #
# ============================= #
# É possível usar qualquer um para gerar as respostas dos agentes. É necessário ter uma chave de API válida para cada modelo.
# O modelo usado no artigo foi o GPT-4o.
# É possível usar um único modelo para todos os agentes, ou usar modelos diferentes entre eles. Mas também não foi explorado no artigo.

model = ChatOpenAI(model="gpt-4o", temperature=1)
#model = ChatOpenAI(model="gpt-4o-mini", temperature=1)
#model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.8,max_tokens=None,timeout=None,max_retries=2,)
#model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=1)

# =================== #
# Criação dos Prompts #
# =================== #
# Inclui instruções específicas e também as informações/contexto relevante de acordo com o pedido do usuário

# Prompt para o agente Helper
def get_helper_prompt(user_query):
    # Instruções estáticas do agente Helper
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
    
    # Retorna o prompt completo
    return f"""Contexto Helper:
    {context}

    {base_instructions}
"""

# Prompt para o agente Data_Analytics
def get_data_analytics_prompt(user_query):
    # Instruções estáticas do agente Data Analytics
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
    
    # Retorna o prompt completo
    return f"""Contexto Data Analytics:
    {relevant_context}

    {base_instructions}
"""

# Prompt para o agente Optimizer
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

    {base_instructions}
"""


# ====================================================== #
# Inicialização dos Agentes com os Prompts e Ferramentas #
# ====================================================== #

# Criação do agente Helper
helper = create_react_agent(
    model=model,
    tools=[queryTools.query],
    name="helper",
    prompt=get_helper_prompt(""),
)

# Criação do agente de Data Analytics
data_analytics = create_react_agent(
    model=model,
    tools=[queryTools.query],
    name="data_analytics",
    prompt=get_data_analytics_prompt(""),
)

# Criação do agente Otimizador
optimizer = create_react_agent(
    model=model,
    tools=[optimizationTools.run_optimization, getPlano.getPlano],
    name="optimizer",
    prompt=get_optimizer_prompt(""),
)

# Criação do supervisor
workflow = create_supervisor(
    [helper, data_analytics, optimizer],
    model=model,
    prompt=(
        "Você é um assistente que gerencia três agentes, Helper, Data Analytics e Otimizador. "
        "Sua tarefa é COMPILAR TODAS AS INFORMAÇÕES fornecidas pelos especialistas em uma única resposta coerente, sem que haja perda de informações. Ou seja, você deve copiar a mensagem encaminhada pelo agente, removendo partes que falem sobre execução de código ou processamento. "
        "Responda usando a mesma língua do usuário. "
        "Ignore qualquer mensagem vazia ou sem conteúdo relevante. "
        "Nunca retorne mensagens vazias. "
        "Responda de forma objetiva, utilizando todo o conteúdo útil recebido. "
        "O nome do usuário é {{username}}, refira-se a ele pelo nome. "
        "No caso de uma primeira interação como 'oi' ou 'bom dia', se apresente ao usuário."
        "Caso contrário, encaminhe a pergunta para o especialista mais adequado: "
        "1 - Para tarefas que envolvam cadastrar ou alterar dados no banco de dados, tirar dúvidas sobre a aplicação e navegação, encaminhe para o Helper. "
        "2 - Para gerar relatórios e gráficos sobre o banco de dados, incluindo consultas de quaiquers dados já cadastados, encaminhe para o Data Analytics. Nesse caso, retorne apenas uma mensagem de confirmação, sem mais detalhes."
        "3 - Para otimizações de carregamento/descarregamento, encaminhe para o Otimizador. "
    ),
)

# Compilação do workflow
app = workflow.compile()

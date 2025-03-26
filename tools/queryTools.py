def query(django_query: str):
    """
    Executa uma query Django ORM no banco da aplicação.
    
    Parâmetros:
    - django_query (str): Código Python representando uma query Django ORM.

    Retorna:
    - Resultado da query, se aplicável.
    """
    try:
        exec(django_query, globals())  # Executa o código Python no contexto global
        return "Query executada com sucesso!"
    except Exception as e:
        return f"Erro ao executar query: {str(e)}"

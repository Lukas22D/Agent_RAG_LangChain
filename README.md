# README.md

Este repositório contém um agente conversacional baseado em langchain que utiliza o modelo de linguagem cohere para responder a perguntas sobre análise, criação e correção de código python, com foco em conceitos relacionados a langchain, bibliotecas e frameworks.

## estrutura do projeto
- src
  - ├── app.py # interface streamlit para interação com o agente
  - └── cohereagent.py # implementação do agente conversacional com langchain

## funcionalidades

- **análise de código**: analisa trechos de código python, identificando erros, problemas de desempenho e áreas para melhoria.
- **criação de código**: gera código python de alta qualidade a partir de descrições ou requisitos, seguindo as melhores práticas.
- **correção de código**: corrige automaticamente erros e problemas em código python existente, fornecendo explicações claras.
- **suporte a langchain**: oferece assistência especializada em tópicos relacionados a langchain, como agentes, cadeias e ferramentas.
- **contexto conversacional**: mantém o contexto da conversa para entender e responder a perguntas de forma mais precisa e relevante.

## tecnologias utilizadas

- **langchain**: framework para desenvolvimento de aplicações com modelos de linguagem.
- **cohere**: modelo de linguagem para processamento de linguagem natural e geração de texto.
- **chroma**: base de dados vetorial para armazenamento e recuperação de informações.
- **streamlit**: biblioteca para criar interfaces web interativas.

## como executar
### Clone o Repositório:
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```
### Instale as Dependências:
```bash
pip install -r requirements.txt
```
### Configure as Variáveis de Ambiente:
```bash
COHERE_API_KEY=sua_chave_api_cohere
```

### Execute o aplicativo:
```bash
streamlit run app.py
```

import os
import ast
from crewai_tools import BaseTool
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
from embedchain import App
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

class SumarizeChain():
    def __init__(self, chain_type="refine"):
        self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        self.chain_type = chain_type

        question_prompt_template = """
                        Forneça um resumo em tópicos dos assuntos tratados no texto a seguir.
                        TEXTO: {text}
                        RESUMO:
                        """

        question_prompt = PromptTemplate(
            template=question_prompt_template, input_variables=["text"]
        )

        refine_prompt_template = """
                    Escreva um resumo conciso do texto a seguir delimitado por crases triplas.
                    Retorne sua resposta em tópicos que cubram todos os pontos-chave do texto.
                    ```{text}```
                    RESUMO DOS PONTOS:
                    """

        refine_prompt = PromptTemplate(
            template=refine_prompt_template, input_variables=["text"]
        )

        if chain_type == "refine":
            self.refine_chain = load_summarize_chain(
                self.llm,
                chain_type="refine",
                question_prompt=question_prompt,
                refine_prompt=refine_prompt,
                return_intermediate_steps=True,
            )
        elif chain_type == "stuff":
            self.refine_chain = load_summarize_chain(
                self.llm,
                chain_type="stuff",
                prompt=question_prompt,
            )

    def summarize(self, document):
        refine_outputs = self.refine_chain({"input_documents": document})
        return refine_outputs["output_text"]

    def summarize_text(self, text):
        document = [Document(page_content=text, metadata={"source": "context"})]
        return self.summarize(document)


class SumarizeVideo(BaseTool):
    name: str = "Resumir Vídeo"
    description: str = (
        "Resume a transcrição de um vídeo do YouTube."
    )

    def _run(self, url_video: str) -> str:
        llm = ChatOpenAI(temperature=0, model_name=model_name)

        loader = YoutubeLoader.from_youtube_url(
            url_video,
            add_video_info=True,
            language=["pt","en"],
        )
        document = loader.load()

        summarize_chain = SumarizeChain()
        refine_outputs = summarize_chain.summarize(document)

        return refine_outputs


class SearchTranscript(BaseTool):
    name: str = "Busca em Transcrição"
    description: str = (
        "Busca o tópico nos vídeos do YouTube e retorna os dados."
    )

    def _run(self, url_video: str, human_input: str) -> str:
        app = App.from_config(config_path="youtube_video.yaml")
        app.add(url_video, data_type="youtube_video")
        results = app.search(f"O que é dito no vídeo sobre: {human_input}", where={"url": url_video})

        output = ""
        output += f"### Tópico: {human_input}\n\n"

        for i in range(len(results)):
            transcript = ast.literal_eval(results[i]['metadata']['transcript'])
            context = results[i]['context']
            #output += f"**Contexto:** {context}\n\n"

            output += f"**Resumo:** {SumarizeChain(chain_type="stuff").summarize_text(text=context)}\n\n"

            # Inicializar variáveis para encontrar o chunk com menor start time
            min_start_time = None
            selected_chunk = None

            # Agrupar chunks de 5 em 5 com 2 sobrepostos
            grouped_chunks = []
            for j in range(0, len(transcript), 2):  # Passo de 3 para sobrepor 2 chunks
                group = transcript[j:j + 5]
                if group:  # Garantir que o grupo não está vazio
                    grouped_chunks.append(group)

            for group in grouped_chunks:
                # Unir os textos dos chunks agrupados
                combined_text = ' '.join(chunk['text'] for chunk in group)

                # Verifica se o texto combinado aparece no contexto
                if combined_text in context:
                    for chunk in group:
                        if min_start_time is None or chunk['start'] < min_start_time:
                            min_start_time = chunk['start']
                            selected_chunk = chunk

            # Se encontrou um chunk correspondente, gera o link
            if selected_chunk:
                segundos = f"{str(selected_chunk['start'])}s"
                title = results[i]['metadata']['title'] + " - " + segundos
                output += f"**Abra o vídeo neste ponto:** [{title}]({url_video}&t={segundos})\n\n"

        return output

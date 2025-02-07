#! /usr/bin/env python
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Sobre este script
-----------------
Este script implementa um processador OCR que utiliza a API Groq
para extrair texto de imagens. Oferece funcionalidades para processar
imagens em faixas, realizar OCR e formatar os resultados em tabelas markdown.
"""
from PIL import Image
from langchain_groq import ChatGroq
import base64
import io
from config.settings import GROQ_API_KEY


class OCRProcessor:
    def __init__(self):
        self.groq_api_key = GROQ_API_KEY

    def encode_image_pil(self, image: Image.Image) -> str:
        """Converte uma imagem PIL em uma string base64 codificada.

        Esta função realiza as seguintes operações:
        1. Converte a imagem para o formato RGB
        2. Comprime a imagem em formato JPEG com qualidade 85
        3. Codifica a imagem comprimida em base64

        Args:
            image (Image.Image): Objeto de imagem PIL para ser codificado.
                            Suporta qualquer modo de cor (será convertido para RGB).

        Returns:
            str: String codificada em base64 da imagem comprimida em formato JPEG.
                Formato: string ASCII contendo caracteres base64 válidos.

        Raises:
            TypeError: Se o parâmetro image não for do tipo PIL.Image.Image
            IOError: Se houver erro durante a conversão ou compressão da imagem
        """
        buffered = io.BytesIO()
        image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def split_image_into_horizontal_stripes(
        self, image: Image.Image, stripe_count: int = 5, overlap: float = 0.1
    ) -> list[Image.Image]:
        """Divide uma imagem em faixas horizontais com sobreposição entre faixas adjacentes.

        Esta função divide uma imagem em várias faixas horizontais de igual altura,
        com uma sobreposição definida entre faixas adjacentes para melhor processamento.

        Args:
            image (Image.Image): Imagem PIL a ser dividida em faixas.
            stripe_count (int): Número de faixas horizontais. Valor padrão: 5.
            overlap (float): Fator de sobreposição entre faixas adjacentes,
                expresso como fração da altura da faixa (0.0 a 1.0). Valor padrão: 0.1 (10%).

        Returns:
            list[Image.Image]: Lista de objetos Image.Image, cada um representando
                uma faixa horizontal da imagem original.

        Raises:
            ValueError: Se stripe_count for menor que 1 ou overlap estiver fora do intervalo [0,1].
            TypeError: Se a imagem não for um objeto PIL.Image.Image.
        """
        width, height = image.size
        stripe_height = height // stripe_count
        overlap_height = int(stripe_height * overlap)

        stripes = []
        for i in range(stripe_count):
            upper = max(i * stripe_height - overlap_height, 0)
            lower = min((i + 1) * stripe_height + overlap_height, height)
            stripe = image.crop((0, upper, width, lower))
            stripes.append(stripe)
        return stripes

    def ocr(
        self, image: Image.Image, model: str = "llama-3.2-90b-vision-preview"
    ) -> str:
        """Realiza reconhecimento óptico de caracteres (OCR) em uma imagem usando um modelo do Groq.

        Esta função processa uma imagem para extrair texto impresso e manuscrito, utilizando
        o modelo de visão computacional da Groq. A imagem é convertida para base64 e enviada
        para processamento através da API Groq.

        Args:
            image (Image.Image): Objeto de imagem PIL contendo o texto a ser extraído.
            model (str): Nome do modelo Groq a ser utilizado.
                        Padrão: "llama-3.2-90b-vision-preview"

        Returns:
            str: Texto extraído da imagem, formatado em português do Brasil (pt-BR).

        Raises:
            ValueError: Se a imagem estiver corrompida ou em formato inválido.
            RuntimeError: Se houver falha na comunicação com a API Groq.
            TypeError: Se o parâmetro image não for do tipo PIL.Image.Image.
        """
        groq_llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=model,
            temperature=0,
            max_retries=3,
        )

        image_data_url = f"data:image/jpeg;base64,{self.encode_image_pil(image)}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                            A imagem contém texto impresso e anotações manuscritas. Sua tarefa é extrair
                            cuidadosamente todo o conteúdo textual, incluindo elementos manuscritos.
                            Sempre retorne o texto em português do Brasil (pt-BR).
                        """,
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ]

        response = groq_llm.invoke(messages)
        return response.content.strip()

    def format_to_table(
        self, markdown_runs: list[str], model: str = "llama-3.3-70b-versatile"
    ) -> str:
        """Consolida e formata múltiplos textos markdown em uma única tabela estruturada.

        Esta função processa múltiplos textos em markdown que podem conter informações
        sobrepostas ou duplicadas, e os consolida em uma única tabela formatada.
        Utiliza o modelo Groq LLM para realizar o processamento e formatação.

        Args:
            markdown_runs (list[str]): Lista de strings contendo textos em formato markdown,
                geralmente obtidos do processamento OCR de diferentes partes de uma imagem.
            model (str): Nome do modelo Groq a ser utilizado para processamento.
                Padrão: "llama-3.3-70b-versatile"

        Returns:
            str: Tabela formatada em markdown, consolidando todas as informações
                sem duplicatas e resolvendo conflitos.

        Raises:
            ValueError: Se a lista markdown_runs estiver vazia.
            RuntimeError: Se houver falha na comunicação com a API Groq.
        """
        groq_llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=model,
            temperature=0,
            max_retries=3,
        )

        combined_markdown = "\n\n".join(markdown_runs)

        messages = [
            {
                "role": "user",
                "content": f"""
                    Você recebeu vários outputs em markdown extraídos de seções sobrepostas de uma imagem.
                    Algumas seções podem conter informações duplicadas ou conflitantes devido à sobreposição.
                    Sua tarefa é:
                    1. Identificar e consolidar linhas de dados que estão relacionadas, garantindo que a versão mais completa da informação seja mantida.
                    2. Para linhas com informações conflitantes (por exemplo, valores diferentes para um campo), priorize a entrada mais detalhada.
                    3. Se um campo está faltando em uma linha mas está presente em outra, combine as informações em uma única linha.
                    4. Produza os dados consolidados em um formato tabular limpo usando a sintaxe Markdown, adequada para renderização direta.
                    5. Somente saída Markdown: Retorne apenas o conteúdo em Markdown sem nenhuma explicação ou comentário adicional.
                    Aqui está o conteúdo a ser processado: {combined_markdown}
                """,
            }
        ]

        response = groq_llm.invoke(messages)
        return response.content.strip()

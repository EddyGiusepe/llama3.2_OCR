#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Execução do script
------------------
streamlit run app.py
"""
import streamlit as st
from PIL import Image
from time import sleep
from src.ocr_processor import OCRProcessor

ocr_processor = OCRProcessor()


# Aplicação Streamlit
st.title("OCR para Tabela com o Modelo Llama3.2 Vision")
st.markdown(
    "Converta o conteúdo de uma imagem carregada em um formato de Tabela Estruturada."
)

# Sidebar para Upload e Exibição:
with st.sidebar:
    st.markdown("#### Upload da Imagem")
    uploaded_file = st.file_uploader(
        "Carregue uma imagem 🖼️", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Redimensiona a imagem para melhor visualização:
        width, height = image.size
        new_width, new_height = int(width * 1.2), int(height * 1.2)
        image = image.resize((new_width, new_height))

        # Exibe a imagem no sidebar:
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Pré-processa a imagem:
        stripes = ocr_processor.split_image_into_horizontal_stripes(image)

# Seção principal para processamento e resultados:
if uploaded_file is not None:
    st.markdown("#### OCR e Resultados")
    progress_bar = st.progress(0)
    n = 1
    markdown_runs = []
    total_steps = len(stripes) * n
    step = 0

    # Caixa de status dinâmica:
    status_box = st.empty()

    for run in range(1, n + 1):
        for i, stripe in enumerate(stripes, start=1):
            step += 1
            progress = step / total_steps
            progress_bar.progress(progress)
            status_box.markdown(
                f"**Processando Stripe {i}, Run {run} ({int(progress * 100)}%) . . . **"
            )
            sleep(0.1)  # Simulando tempo de processamento

            stripe_markdown = ocr_processor.ocr(
                stripe, model="llama-3.2-90b-vision-preview"
            )
            markdown_runs.append(stripe_markdown)

    progress_bar.progress(1.0)
    status_box.markdown("**Processamento completo.**")

    # Exibindo os resultados:
    table_output = ocr_processor.format_to_table(
        markdown_runs, model="llama-3.3-70b-versatile"
    )
    st.markdown(table_output, unsafe_allow_html=True)

    # Botão de download:
    st.download_button(
        label="Download Tabela",
        data=table_output.encode("utf-8"),
        file_name="tabela_consolidada.txt",
        mime="text/plain",
    )
else:
    st.markdown("O resultado será exibido aqui após carregar uma imagem.")

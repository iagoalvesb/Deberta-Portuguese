# [EM PRODUÇÃO]

# Deberta-Portuguese

Este projeto, baseado no modelo [Deberta](https://github.com/microsoft/DeBERTa) de código aberto da Microsoft, é pré-treinado no domínio português.

Este modelo foi pré-treinado com base na porção em portugues do corpus [OSCAR](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) (Open Super-large Crawled Aggregated coRpus). O projeto OSCAR é um projeto de código aberto que se concentra especificamente no fornecimento de grandes quantidades de recursos e conjuntos de dados multilíngues baseados na Web, dados brutos não anotados que são comumente usados no pré-treinamento de grandes modelos de aprendizado profundo. A porção do corpus que corresponde à língua portuguesa dispõe de 105.0 GB de dados textuais, sendo esse o conjunto utilizado para o treinamento do modelo.

O [tokenizador](https://huggingface.co/iagoalves/portuguese_deberta_tokenizer) foi treinado em 1% do corpus citado e possui o tamanho de vocabulário igual a 128k.

## Reprodução

O pré-treino do modelo Deberta no domínio da língua portuguesa se deu a partir da task Replaced Token Detection (RTD), seguindo a lista de passos destacados no repositório base. A forma de realizá-lo foi por meio do [Docker](https://hub.docker.com/r/bagai/deberta) disponibilizado, necessitando realizar a instalação do package da deberta e alterar para se adequar ao novo tokenizer.

Entretanto, durante o processo alguns erros foram levantados, sendo o principal o apontamento de que as tasks Masked Language Modeling (MLM) e Replaced Token Detection (RTD) não estão registradas nas tasks disponíveis.

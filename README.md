Projeto para rodar a implementação da dissertação de mestrado

Texto da dissertação e pdf da defesa em docs/

# Fluxo de execução
* Prepração:
  * Descompactar arquivos CoNLL
  * Descompactar modelo SpanBERT
* Encoding:
    * rodar create_bert_embeddings.py para criar a versão codificada dos arquivos conll
    * rodar create_bins_files.py para criar bins
  
## Docker
Configurar arquivo docker/.env apontando as pastas correspondentes
Rodar a partir da pasta docker com
<pre>
  docker-compose up [comando]
</pre>
Usar um dos seguintes comandos:
* encode-conll : Cria os encodings e bins dos arquivos conll
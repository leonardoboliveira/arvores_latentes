Projeto para rodar a implementação da dissertação de mestrado

Texto da dissertação e pdf da defesa em docs/

# Fluxo de execução
* Prepração:
  * Descompactar arquivos CoNLL
  * Descompactar modelo SpanBERT
* Encoding:
    * rodar create_bert_embeddings.py para criar a versão codificada dos arquivos conll
    * rodar create_bins_files.py para criar bins
* Treino:
  * Criar Batch files
  * Alterar scripts/flags.sh : Nesse arquivo estão os hyperparametros do modelo
  * Alterar extra_files/features.txt: Nesse arquivo estao as features q serao usadas no encoding de arcos
  * Treinar modelo
  
## Docker
Configurar arquivo docker/.env apontando as pastas correspondentes
Rodar a partir da pasta docker com
<pre>
  docker-compose up [comando]
</pre>
Usar um dos seguintes comandos:
* encode-conll: Cria os encodings e bins dos arquivos conll
* create-batches: Cria os arquivos de batch. Para melhorar a performance o dataset da 
  CoNLL precisa ser dividos em batches. O tamanha do batch vai depender de quanta memória se tem disponível
  e qual o tamanho do encoding
* train: realiza o treino e faz uma mini-validação a cada 6 batches
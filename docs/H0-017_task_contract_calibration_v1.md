# H0-017: contrato explícito e calibração textual

Autorizado em 08/10/2026 após a rodada funcional: esclarecer o contrato de
correção e calibrar a referência textual antes de escolher outra intervenção
de treino. Core + name continuam uma unidade. Este ensaio não treina modelos,
não avalia comunicação latente e não altera o gate histórico nem seus resultados.

## O que muda

O registro `config/H0-017_task_contract_v2.json` conserva as 32 tarefas, suas
especificações originais, hashes e três testes originais. Acrescenta assinatura,
domínio de entrada e retorno observável. O código de referência do MBPP foi
lido por AST, sem execução local, na revisão congelada
`4bb6404fdc6cacfda99d4ac4205087b89d32030c`, configuração full, validation.
O Parquet tem SHA-256
`3f0ec060987432d99fe8fb409d31e6c67445b208a01741c5583517c80a10fe80`.

Exemplos: `is_Sub_Array(A,B,n,m)` informa o papel dos comprimentos;
`last_occurence_char` informa posição começando em 1 ou None;
`volume_tetrahedron(num)` informa tetraedro regular e arredondamento em duas
casas; `Seq_Linear` informa as duas strings de retorno exigidas.

O prompt público é construído somente com campos explicitamente selecionados.
Ele não recebe casos de teste, saídas esperadas, código de referência ou
respostas anteriores. A revisão foi feita depois de observar falhas de
desenvolvimento: é uma intervenção pós-hoc declarada, não uma avaliação cega.

## Divergências que impedem chamar o conjunto de benchmark limpo

| Tarefa | Evidência e decisão explícita |
|---|---|
| 516 | Referência de radix sort interrompe cedo em algumas potências de dez. Contrato exige ordenação correta; não incorpora o bug. |
| 528 | Referência escolhe a menor lista lexicográfica, não necessariamente a mais curta. Contrato segue menor comprimento, sem empate no domínio. |
| 540 | Referência não finaliza a última sequência de valores iguais. Contrato segue diferença entre frequências. |
| 558 | “Digit distance” é indefinido; testes não distinguem duas leituras. Contrato escolhe soma dos dígitos da diferença absoluta, como a referência. |
| 582 | Enunciado diz dicionário; dois testes passam conjuntos. Novo domínio declara ambos. |
| 584 | Enunciado pede todos os advérbios; referência retorna só o primeiro. Novo contrato declara primeiro match e formato exato. |
| 592 | Enunciado não define quais produtos de coeficientes somar. Contrato define produtos adjacentes na linha n de Pascal, consistente com a referência. |

As sete são identificadas antes de gerar novas respostas. Todas permanecem no
denominador 32. Relatar também sete sinalizadas e 25 restantes, sem escolher
subconjuntos pelo acerto novo. “Restantes” não significa prova de correção ou
cobertura completa. Referências podem passar os três testes e ainda ter bugs.

## Calibração pareada

Mesma revisão do Llama-3-8B-Instruct, quantização de 4 bits, L4, geração greedy,
seed 4513 e limite de 256 novos tokens. Mesmo system prompt e template. Gerar
uma resposta por tarefa em cada condição, sem seleção posterior:

1. `text_original`: enunciado e nome originais, com paridade dos tokens de
   entrada verificada contra a rodada anterior.
2. `text_signature`: o mesmo prompt acrescido apenas da assinatura.
3. `text_explicit`: contrato público completo, inclusive revisões semânticas
   declaradas acima. Não é uma intervenção apenas de formatação.

São 96 respostas novas. Outras 32 linhas contêm o código de referência
congelado para verificar o funcionamento do avaliador no mesmo sandbox. Estas
não são respostas geradas nem entram em taxas de acerto do modelo.

Repetir a condição original na sessão atual controla diferenças de ambiente;
registrar também coincidência token a token com a resposta antiga. Medir ganhos
e perdas pareados entre condições, limites de tokens, erros e incompatibilidades
estáticas de chamada. Um TypeError no corpo não equivale a erro de aridade.

Fixar Transformers 5.16.1, Accelerate 1.14.0 e bitsandbytes 0.50.2 com
`requirements-h0-017-contract-calibration.txt`. A primeira tentativa em
08/10/2026 encontrou Transformers 5.18.0 sem `config._commit_hash` e parou
antes de gerar respostas. Seu log foi preservado; não se removeu a verificação
de revisão. Torch/CUDA são registrados por execução e a repetição textual
controla também possíveis diferenças em relação ao ambiente de setembro.

## Contrato de correção

A métrica primária permanece passar todos os testes originais sem reparo.
Não renomear funções, adaptar argumentos, mudar corpos, afrouxar asserts ou
adicionar casos depois de ver respostas. A verificação estática de assinatura
é diagnóstica: decoradores e aliases podem torná-la inconclusiva. Ela nunca
anula uma execução funcional aprovada. Igualdade e tolerância são exatamente
as dos asserts originais, sem nova exigência de tipo ou identidade do objeto.

Algoritmos solicitados (radix sort, map, heap, operador bitwise), efeitos de
mutação e comportamento fora dos casos existentes não são certificados pelos
três asserts. O contrato explicita intenção; aprovação mede somente a cobertura
disponível. Não é prova de solução universal. Novos testes demandam outra versão
congelada antes da geração, não alteração retroativa desta rodada.

Toda execução de código de candidato e referência passa pelo sandbox Linux
com namespaces validados, sem rede, sem acesso ao Drive e sob usuário sem
privilégios. Evidências e saídas são persistidas antes da correção. Sem sandbox
validado, a avaliação falha fechada.

## Consequência para os pacotes

Um prompt diferente gera estados diferentes. A calibração usa somente texto;
não recicla os pacotes antigos como se contivessem o contrato novo. Uma futura
comparação latente terá de extrair novamente emissor e receptor usando o mesmo
contrato, manter core + name juntos e incluir controles matched/shuffled. Dar
a especificação apenas ao receptor criaria outro canal de informação.

Comando de validação local (sem gerar ou executar candidatos):

```text
python -m src.scripts.run_task_contract_calibration --references selected_references.json --output contract-calibration-v1 --validate-only
```

No Colab, retirar `--validate-only` para geração e usar
`src.scripts.run_hardened_oracle_evaluation` com a política desta calibração
para correção. Preparação e testes locais não são resultado experimental.

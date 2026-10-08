# H0-017: resultado da calibração do contrato — 08/10/2026

O contrato completo elevou a referência textual de 10/32 para 22/32 acertos
funcionais nas mesmas 32 tarefas de desenvolvimento. Não houve treino nem
replay latente. Os três testes originais de cada tarefa foram mantidos, sem
reparar nomes, argumentos, imports ou corpos dos candidatos.

| Informação pública | Passa todos os testes | Incompatibilidade de chamada | Sintaxe válida | Limite de tokens |
|---|---:|---:|---:|---:|
| Original | 10/32 (31,25%) | 9 | 31 | 1 |
| Original + assinatura | 14/32 (43,75%) | 0 | 31 | 1 |
| Contrato completo | 22/32 (68,75%) | 0 | 32 | 0 |
| Código de referência MBPP, não gerado | 32/32 | 0 | 32 | 0 |

A condição original reproduziu 32/32 saídas antigas token a token. Receptor,
revisão, quantização e decodificação foram mantidos. Torch/CUDA da sessão são
registrados na metadata; Transformers/Accelerate/bitsandbytes foram fixados
nas versões anteriores após detectar incompatibilidade de metadados na 5.18.0.

Pareamento original → contrato completo: 14 ganhos e duas perdas (591 e 573),
saldo de 12 tarefas. Original → assinatura: cinco ganhos e uma perda (558).
Assinatura → completo: 11 ganhos e três perdas (591, 545 e 573). Portanto,
explicitar informação melhora o agregado, mas não monotonicamente cada tarefa.

As sete tarefas sinalizadas antes da geração foram mantidas: 516, 592, 540,
584, 582, 528 e 558. Nelas, os acertos original/assinatura/completo foram
2/7, 2/7 e 6/7. Nas 25 restantes, foram 8/25, 12/25 e 16/25. O ganho não
depende exclusivamente dos sete casos com ressalvas.

## Dez falhas com contrato completo

| ID | Evidência no código gerado |
|---|---|
| 580 | Remove níveis de tuplas ao concatenar a chamada recursiva. |
| 582 | Retorna `bool(dict1)`, invertendo o sentido de vazio. |
| 513 | Acrescenta a string apenas ao fim, em vez de após cada elemento. |
| 519 | Usa volume `num**3 / 3`, fórmula incorreta para tetraedro regular. |
| 544 | Converte tuplas inteiras para string, sem achatar seus elementos. |
| 591 | Gira a lista em vez de trocar somente primeiro e último elementos. |
| 586 | Inverte a ordem interna das duas partes do array. |
| 524 | A recorrência não impõe subsequência estritamente crescente. |
| 545 | A expressão de bits não implementa o toggle solicitado. |
| 573 | Chama `math.prod` sem importar `math`. |

São nove AssertionError e um NameError. Nenhuma falha restante é de aridade,
sintaxe ou truncamento. Não foi feita correção posterior desses candidatos.

## Interpretação e continuação

A informação pública da tarefa era uma fonte relevante de falha do controle
textual. Ainda há erros de solução e sensibilidade ao prompt. Algumas mudanças
do contrato escolhem uma interpretação semântica; este experimento não isola
formatação. Os três asserts disponíveis também não certificam algoritmo,
comportamento fora dos casos, nem corrigem os defeitos das referências.

Este resultado não mede melhoria do canal latente, generalização ou economia.
Core + name permanecem uma unidade. O próximo diagnóstico sugerido é reextrair
os pacotes sob o contrato congelado e comparar texto e pacote oráculo, incluindo
matched/shuffled. Não reutilizar pacotes antigos como se contivessem o contrato
novo. Não exigir 32/32 como condição para quantificar a perda adicional do canal.

## Proveniência

- Geração: `77e1289aeb7de777b908745aaa2a17feebff6611`.
- Correção final: `78f17300d2f2e9551378c89436fbcd9ddf874dc0`.
- Respostas SHA-256: `95f219a9b1f0afddfbaeb57588b07670c47e4abc172b1dfb9a0d0a3ad32b3242`.
- Scored SHA-256: `405c3848897ce2ae672ededca448d884232cdb9b4557357329c5c85af51a980c`.
- Summary SHA-256: `8a6107ddace4c8a33f920694d860c893febeeb9dd5b078e0c7bead36bc4add1e`.
- Sandbox SHA-256: `c481c955fefd72b9d53edea38b78c3bea7395df30f13d1615e21e35322782989`.
- 26 testes de contrato/sandbox passaram localmente e no Colab após a correção
  da integração do resumo. As tentativas interrompidas foram preservadas.
- A primeira tentativa não gerou respostas devido à incompatibilidade de
  metadados do Transformers. Depois, campos de resumo exigidos pelo executor
  foram acrescentados. A correção foi repetida sobre as mesmas respostas;
  os acertos e comparações pareadas permaneceram iguais.
- Execução dos candidatos em sandbox Linux validado, sem rede nem acesso ao
  Drive. Persistência no Drive conferida por tamanho e MD5, com SHA-256 registrado.

[Resultados, contratos e arquivos completos no Drive](https://drive.google.com/drive/folders/1AsUOx7YVjwirR0ym_F9o1amv55pj2_0l).

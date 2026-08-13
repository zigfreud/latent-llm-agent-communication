# LIP-H0-013 — core-margin pressure screen

H0-012 established a reproducible partial property: with the receiver evolution
and example exposure fixed, increasing the number of in-batch alternatives from
three to fifteen made the core margin positive in all three seeds. It did not
make the core Holm test significant in any seed.

H0-013 changes exactly one training scalar relative to the frozen H0-011 seed
4007 screen: the induced-trajectory `lambda_margin` rises from 0.10 to 1.00.
The entry objective remains unchanged. Architecture, receiver, unrolled layers,
data, batch size, update count, total examples, seed, NCE, reconstruction terms,
margin target, selection rule, and development gate all remain fixed.

The tenfold value is not an arbitrary architectural search. In the frozen
H0-011 history, final regional margin losses were 0.00330 for joint, 0.022997
for core, and zero for name. Core was therefore the only persistent regional
margin violation, while the aggregate margin term had only 0.10 weight. Raising
that scalar supplies pressure to the observed active deficit without inventing
a new loss family.

The screen is paired to the existing H0-011 seed 4007 cell. Strong success
requires improvement without lower core or mean retrieval and passage of the
complete joint/core/name Holm family. Only strong success authorizes exact
replication on seeds 4001 and 4003. Confirmation data and PROTO-015 remain out
of scope.

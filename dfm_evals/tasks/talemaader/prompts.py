JUDGE_TEMPLATE_DA = """\
Du vurderer et indsendt svar mod et ekspertsvar om en dansk talemåde.

[BEGIN DATA]
************
[Talemåde]: {talemaade_udtryk}
************
[Facit]: {criterion}
************
[Modelsvar]: {answer}
************
[END DATA]

Sammenlign det semantiske indhold i modelsvaret med facit. \
Ignorer forskelle i stil, grammatik eller tegnsætning.

I denne opgave skal modelsvaret faktisk forklare, hvad talemåden betyder. \
Det er ikke nok blot at gentage talemåden, omskrive den næsten ordret eller \
bruge den i en ny sætning. Svar kun positivt, hvis modelsvaret tydeligt \
udtrykker selve betydningen eller definitionen af talemåden med egne ord.

Indeholder modelsvaret den korrekte betydning af talemåden?

{instructions}
"""

JUDGE_INSTRUCTIONS_DA = (
    "Forklar først kort og trin for trin, hvordan modelsvaret stemmer overens "
    "med facit. Skriv derefter præcis én afsluttende linje med formatet "
    "'GRADE: $LETTER' hvor LETTER er en af CPI. Brug ikke 'GRADE:' tidligere "
    "i svaret, og placer grade-linjen som den allersidste linje. Et svar er "
    "kun korrekt, hvis det faktisk forklarer betydningen af talemåden. Det er "
    "ikke nok at gentage udtrykket, omskrive det næsten ordret eller give en "
    "cirkulær parafrase uden at skrive, hvad talemåden betyder. "
    'Vælg én mulighed: "C" for korrekte svar, '
    '"P" for delvist korrekte svar, '
    'eller "I" for forkerte svar.'
)

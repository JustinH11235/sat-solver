# Project 1 Interim

Justin Hinckley

jhinckley6@gatech.edu

903577463

## Honor Code
I confirm I adhered to the honor code.

## Encoding
Look in `generate_riddle_cnf()` for my script that generates clauses for the riddle.
Look in riddle.cnf for the formulas in CNF form.

### Propositions
My propositions are in the form $H_{i,PROPERTY} :=$`((HOUSE #, PROPERTY), NEGATED)`. For example, `((3, milk), -1)` means $\lnot$ (House #3 drinks milk). Whereas `((3, milk), 1)` means (House #3 drinks milk). The houses are numbered [1, 5] inclusive and house 1 is the leftmost house whereas house 5 is the rightmost house. All properties are related to a House #.

### Constraints

#### Constraint 1 (Each property value (e.g. milk) must be assigned to at least one House #)


#### Constraint 2 (Each House can have at most 1 color, 1 nationality, 1 cigar, 1 pet, and 1 beverage)

Together, these 2 constraints encode the riddle constraints, because since we have 5 houses and 5 options per property, if each house can't have more than 1 of a property and every property must get assigned, implictly, each house must be assigned exactly 1 value for each property, otherwise we wouldn't be able to cover all 25 properties.

### Example Hint Encodings
Most of the hints follow the form: `PROP1` relates to `PROP2`. To encode this, I create a disjunction of conjunctions like so: $\bigvee_{H_i \in HOUSES} H_{i,PROP1} \land H_{i,PROP2}$, meaning there's some house that has both properties. Then I have to convert it to CNF form so I convert the disjunctions to conjunctions by distributing the disjunction across the conjunctions:

For example this:
$(H_{1,PROP1} \land H_{1,PROP2}) \lor (H_{2,PROP1} \land (H_{2,PROP2}) \lor (H_{3,PROP1} \land H_{3, PROP2})$

Would become this:
$(H_{1,\text{PROP1}} \lor H_{2,\text{PROP1}} \lor H_{3,\text{PROP1}}) \land (H_{1,\text{PROP1}} \lor H_{2,\text{PROP2}} \lor H_{3,\text{PROP1}}) \land (H_{1,\text{PROP1}} \lor H_{2,\text{PROP2}} \lor H_{3,\text{PROP2}}) \land (H_{1,\text{PROP2}} \lor H_{2,\text{PROP1}} \lor H_{3,\text{PROP1}}) \land (H_{1,\text{PROP2}} \lor H_{2,\text{PROP2}} \lor H_{3,\text{PROP1}}) \land (H_{1,\text{PROP2}} \lor H_{2,\text{PROP2}} \lor H_{3,\text{PROP2}})$

Some hints relate the positions of houses, such as "the green house is left of the white house", which I encode as $\bigvee_{H_i \in HOUSES} H_{i, Green} \land H_{i+1, White}$. I then convert to CNF form also.

Other hints relate the positions of houses but state that the houses can be on either side of each other like "Blends live next to house with cats", which I encode as and then convert to CNF.

$\bigvee_{H_i \in HOUSES} (H_{i, Green} \land H_{i+1, White}) \lor \bigvee_{H_i \in HOUSES} (H_{i, White} \land H_{i+1, Green})$

The remaining hints are simpler, such as "The Norwegian lives in the first house" := $H_{1,Norwegian}$

### Interpreting Clauses
To interpret the Clauses outputted by my program, my formuals have the type:

`FORMULA: List<CLAUSE>`

`CLAUSE: List<PROP>`

Where `CLAUSE`s are combined with the $\land$ operation, and the propositions within a clause are combined with the $\lor$ operation.

## Assignments
The below assignments are assigned to True:

[((1, 'cat'), 1), ((1, 'dunhill'), 1), ((1, 'norwegian'), 1), ((1, 'water'), 1), ((1, 'yellow'), 1), ((2, 'blend'), 1), ((2, 'blue'), 1), ((2, 'danish'), 1), ((2, 'horse'), 1), ((2, 'tea'), 1), ((3, 'bird'), 1), ((3, 'british'), 1), ((3, 'milk'), 1), ((3, 'pallmall'), 1), ((3, 'red'), 1), ((4, 'coffee'), 1), ((4, 'fish'), 1), ((4, 'german'), 1), ((4, 'green'), 1), ((4, 'prince'), 1), ((5, 'beer'), 1), ((5, 'bluemaster'), 1), ((5, 'dog'), 1), ((5, 'swedish'), 1), ((5, 'white'), 1)]

Every other proposition is False.

## Solution
[4, 'german', 'prince', 'green', 'coffee', 'fish']

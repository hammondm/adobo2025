# ADOBO 2025 shared task

This repo has my materials for the shared task on automatic detection of borrowings at IberLEF 2025.

## Overview

The general strategy here was to build a simple system that uses only tools that were discussed in my recent course on *statistical natural language processing* at the U. of Arizona (Spring 2025).

https://faculty.sbs.arizona.edu/hammond/ling439539-sp25/

Final results are not yet in, but my hope is that these simple tools can produce a system that is not embarrassingly bad.

## The task

The task was to identify borrowings from English in sentences of Spanish. The task was interesting in that no training data were provided and participants were allowed to use any resources they wanted to build their systems.

Here is a description of the task:

https://adobo-task.github.io/borrowing.html

Here is the submission website:

https://www.codabench.org/competitions/7284/

## Sample data

Sample data for the task appear in the `reference.csv` file provided on the last website above. Here are the first 10 lines of that file:

```verbatim
Somos un país en el que 'youtubers' y 'gamers' millonarios deciden irse al paraíso fiscal de Andorra porque tributan menos sin preocuparse del bienestar de sus vecinos y de quienes les han hecho ricos, ni acordarse de la Educación, Sanidad, infraestructuras de las que han disfrutado durante años gracias a la solidaridad de todos.;youtubers;gamers;;;
Youtubers La polvareda levantada en medios de comunicación y redes sociales por la marcha a Andorra de algunos 'youtubers' millonarios para ahorrarse impuestos reproduce polémicas similares de hace años protagonizadas por célebres cantantes, actores y deportistas.;Youtubers;youtubers;;;
Teniendo en cuenta los millones de seguidores con los que cuentan ya algunos de los youtuber o instagramers que ofrecen contenidos de este tipo, si algo está claro en el universo ASMR es que ha llegado para quedarse.;youtuber;instagramers;;;
"Pocos definieron mejor la esencia del capitalismo que John Steinbeck en ""Las uvas de la ira"", cuando en el Crac del 29, un yankee neto, desahuciado de sus tierras, amenaza con presionar con su rifle al responsable pero en la cadena del daño no encuentra quién es.";yankee;;;;
Sacha Baron Cohen tiene también un personaje totalmente políticamente correcto, progresista, que se siente mujer y hombre, todas esas cosa que definen a estos movimientos 'woke' (despiertos socialmente) que son absolutamente absurdos, un animalismo totalmente tonto, que existe más en Estados Unidos que en España.;woke;;;;
"Podemos está exigiendo acabar con los ""beneficios caídos del cielo"" de determinadas tecnologías (también conocidos como 'windfall profits') que el acuerdo de Gobierno prometía abordar.";windfall profits;;;;
"No se menciona al Doctor Li Wenliang, el oftalmólogo conocido por muchos como el whistleblower tras haber sido amonestado por la policía por advertir a sus colegas sobre una ""enfermedad parecida al SARS"" el 30 de diciembre del año pasado y que más tarde murió a causa de la COVID-19.";whistleblower;;;;
'Whataboutism' o cómo la derecha española usa el asalto al Capitolio para ajustar cuentas con la izquierda En Estados Unidos, lo llaman 'whataboutism' y se ha convertido en una herramienta recurrente en la pelea propagandística.;Whataboutism;whataboutism;;;
Con el móvil apuntando al teclado Para evitar los fraudes, los profesores están optando principalmente por dos vías: la videovigilancia a través de las webcam y / o los móviles de los estudiantes (en ocasiones ambas a la vez) y adaptar los exámenes a la modalidad a distancia.;webcam;;;;
Antes 79,95 € y ahora por 51,16 €.con un estampado divertido y colorido, gracias a sus dos capas de tejido, su cierre con velcro en los puños y con total impermeabilidad waterproof, resistencia y transpirabilidad irán cómodas muy abrigadas.;waterproof;;;;
```

Fields are separated with `;`. The sentence appears first and borrowings occupy remaining fields on the line.

Testing was on the basis of the `input.csv` file, also on the last website above. This file also has a single sentence on each line. Here are the first 10 lines:

``` verbatim
"""Burpees"" para perder kilos sin salir de casa"
"""Cool"", cómoda y versátil: la gabardina que está de moda esta temporada"
"""Pipeline"" de canalización de datos de código abierto para almacenar y extraer información"
"""Dealer"" de estupefacientes detenido en una operación antidroga"
"""Eyeliner"" a todo color para para un verano fantástico"
"""Frame"" de la nueva serie ""Big little lies"" de Netflix"
"""Hate"", falta de privacidad y ciberdelincuencia: la cara oculta de las redes"
"""Hoodie"" con capucha, botines y gafas de pasta conforman el atuendo que le han hecho famoso."
"""Revival"" de los ochenta en los carnavales de este año"
"""Gamer"", mujer y homosexual: la nueva estrella del partido republicano"
```

The goal was to extract the borrowings from the sentence and append them as additional fields on the line.

One interesting aspect of the task is that a borrowing can be a sequence of words. Thus, in line 6 above, *Big little lies* is the borrowing, rather than the three words separately.

## My approach

I ended up trying three different kinds of systems.

1. logistic regression
1. simple neural net
1. simple neural net + rules

The last system performed best and the code files included here are for that system.

The different systems all made use of these resources:

1. COALAS dataset (https://github.com/lirondos/coalas). These data were used for training all models.
1. NLTK wordlists
    - `nltk.corpus.words`
    - `nltk.corpus.cess_esp`
1. NLTK stemmers
    - `nltk.stem.PorterStemmer`
    - `nltk.stem.SnowballStemmer`
1. unix/mac English wordlist (`/usr/share/dict/words`)
1. Another English wordlist (https://github.com/dwyl/english-words)
1. `spacy` taggers for English and Spanish

The COALAS dataset is organized by word. Here's the first few lines of the training partition.

```verbatim
Microsoft       O
promete O
formación       O
digital O
gratis  O
a       O
25      O
millones        O
de      O
personas        O
este    O
año     O

El      O
gigante O
del     O
software        B-ENG
Microsoft       O
lanzó   O
este    O
martes  O
```

Each word appears on its own line and is coded for whether it's a borrowing or not. Sentences are separated by spaces.

### Logistic regression

I set this up as binary logistic regression where 0 is Spanish and 1 is borrowed.

I generated a set of features based entirely on intuition. These include:

1. Is the word in all capitals?
1. Is the first letter capitalized?
1. Does the word include non-alphabetic characters?
1. Does the word include common Spanish suffixes?
1. Does the word appear in any of the English wordlists?
1. Does the word appear in the Spanish wordlist?
1. What is the character bigram probability with respect to English?
1. What is the character bigram probability with respect to Spanish?
1. Can the word be stemmed for English?
1. Can the word be stemmed for Spanish?
1. I tagged the sentence for Spanish with `spacy` and generated a rule for each tag.

With those rules in hand, Each item in the COALAS dataset was used for training. Each item was scored with respect to the rules above and then converted to *z*-scores.

Using the `reference.csv` file for testing, this got an $F_1$ around $.6$.

### Simple neural net

The logistic regression model did not include interactions between factors, so I implemented a neural net with `pytorch`. The architecture was extremely simple: 3 layers with the same dimensions as the input. There was a final output layer that produced a single output value. The activation function for all layers was sigmoid.

Again testing with `reference.csv` and various numbers of epochs and different batch sizes, this reached $F_1$ values as high as $.65$.

### Simple neural net + rules

I augmented the neural net with a rule-based system. That is, I applied rules to the output of the network. Specifically:

1. All quoted sequences of up to 3 words are borrowings.
1. All capitalized sequences of up to 3 words are borrowings.
1. If a word appears in any of the English wordlists and does not appear in the Spanish wordlist, then it's a borrowing.

This approach reached an $F_1$ of about $.74$.

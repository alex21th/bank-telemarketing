## Enfocament basat en dades per predir l'èxit del telemarketing bancari

> Aprenentatge Automàtic 1 - (Machine Learning 1)
>
> David Bergés; Alex Carrillo; Roser Cantenys Sabà
>
> Primavera 2019

### Introducció

En aquest treball s'estudia la venta de dipòsits bancaris a través de trucades de telemàrqueting. Durant aquestes campanyes, els agents realitzen trucades telefòniques a una llista de clients per vendre el producte financer (_outbound_) o, altrament, el client truca al centre de contacte per algun motiu aliè a la campanya i se l'informa sobre la subscripció del dipòsit (_inbound_). El resultat, així doncs, és un diàleg amb èxit o no de la venda.

### Objectiu del treball

Aquest projecte té diversos objectius. El primer, de caire més general, es basa en la recerca i investigació de les estratègies de les campanyes de comercialització de vendes i la segmentació de clients per assolir un objectiu específic de negoci. És a dir, es vol estudiar la influència dels diferents paràmetres a l’hora de comprar o no un dipòsit bancari.

El segon objectiu consisteix en la teoria portada a la pràctica. Això és replantejar-se el màrqueting centrant-se en minimitzar el nombre de trucades (_outbound_) maximitzant el nombre de clients que contracten el dipòsit bancari. En aquest projecte, es vol potenciar la tasca de seleccionar el millor conjunt de clients, el més probable de subscriure's a un producte.

### Dades

L'estudi considera dades reals recollides d'una institució bancaria portuguesa de maig de 2008 al juny de 2013, amb un total de 41.188 contactes telefònics. Cada registre inclou la variable objectiu de sortida amb el resultat del contacte (`{"fracàs", "èxit"}`) i les característiques d'entrada dels candidats.

Aquestes inclouen atributs de telemàrqueting (_p. ex._ el mitjà de comunicació), detalls del producte (_p. ex._ el tipus d'interès ofert) i informació del client (_p. ex._ l'edat). A més, la llista conté certa informació d'influència social i econòmica (_p. ex._ la taxa de variació de l'atur) extreta de la web del Banc Central de la República Portuguesa ([aquí](https://archive.ics.uci.edu/ml/datasets/bank+marketing)). La fusió d'ambdues fonts de dades constitueix un gran conjunt de característiques potencialment útils, amb un total de 21 atributs, que es revisen a continuació. [...]
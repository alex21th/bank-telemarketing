library(ggplot2)
library(caret)
library(caretEnsemble)
library(ROSE)
library(mlbench)
library(DMwR)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)


#Primer de tot, llegim el nostre fitxer de dades
dd <- read.csv('bank-additional-full.csv', sep = ';', header = TRUE, na.strings = c('unknown'))

N <- dim(dd)[1] #nombre total de dades de la notra database

# Anem a analitzar les diferents variables del nostre model
str(dd)

# Primer analitzem la nostra variable resposta; a veure si s'ha fet un dipòsit o no mitjançant una campanya publicitària basada en trucades telefòniques.
response <- table(dd$y)
response
response / N
# Com veiem tenim un 89% de les nostres dades que no fan un dipòsit, i només un 11% que sí que el fa.
# A priori podríem dir que serà difícil obtenir un model de predicció molt bo.
bar1 <- ggplot(dd, aes(x = y, fill = y)) +
        geom_bar(color = 'black') + 
        scale_fill_brewer(palette = 'Spectral') + 
        ggtitle('Deposits') + labs(x = 'Success?', y = 'Number of clients') + 
        theme(plot.title = element_text(hjust = 0.5), legend.position = 'none')
bar1

# Variable 'age'
summary(dd$age) #L'edat mínima és 17 i la màxima 98, el que semblen valors raonables.
hist(dd$age, col = 'light blue', main = 'Age distribution', xlab = 'Age in years', ylab = 'Number of people')
# Intentem veure a priori si la variable age té cap efecte a la nostra variable resposta.
boxplot_age <- ggplot(dd, aes(y, age)) + 
               geom_boxplot(aes(fill = y)) + 
               ggtitle('Deposits vs Age') + 
               labs(x = 'Success?', y = 'Number of clients') + 
               theme(plot.title = element_text(hjust = 0.5), legend.position = 'none')
boxplot_age #A priori sembla que l'edat no sigui una variable que influeixi molt pel que fa a la classificació de la nostra variable resposta.

# Variable 'job'
summary(dd$job) #Tenim 330 persones de les quals no sabem la seva feina
bar2 <- ggplot(dd, aes(x = job, fill = job)) + 
        geom_bar(color = 'black') + 
        scale_fill_brewer(palette = 'Spectral') + 
        ggtitle('Jobs') + 
        labs(x = '', y = 'Number of people') + 
        theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90, hjust = 1), legend.position = 'none')
bar2
bar3 <- ggplot(dd, aes(x = job, fill = y)) + 
        geom_bar(color = 'black') + 
        scale_fill_brewer(palette = 'Spectral') + 
        ggtitle('Depostits per Jobs') + 
        labs(x = '', y = 'Number of people') + 
        theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90, hjust = 1)) +
        guides(fill = guide_legend(title = 'Success'))
bar3
# Sembla que a partir d'aquest gràfic no podem extreure conclusions, anem a comparar els percentatges de gent que sí fa un dipòsit depenent de la seva feina.
jobs <- table(dd$job, dd$y)
jobs
jobs.freq <- jobs / rowSums(jobs)
jobs.freq
# En un principi sembla que el percentatge d'èxit en grups com els 'retired' i els 'student' és més alt. Anem a comprobar-ho gràficament.
jobs.freq.plot <- data.frame(jobs.freq[,2]) #ggplot requires a dataframe to plot something
names(jobs.freq.plot) <- c('success')
bar4 <- ggplot(jobs.freq.plot, aes(x = rownames(jobs.freq.plot), y = success, fill = rownames(jobs.freq.plot))) +
        geom_bar(stat = 'identity', color = 'black') +
        scale_fill_brewer(palette = 'Spectral') +
        ggtitle('% of campaign success per job') +
        labs(x = '', y = '% of yes') +
        theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90, hjust = 1), legend.position = 'none')
bar4
# Efectivament, tal i com podem veure en el gràfic, sembla que el fet de ser retirat o estudiant influeix en la nostra variable resposta. Així que com a intuïció prèvia, podríem dir que els bancs pòtser estarien interessats en oferir els dipòsits a més persones que encaixin en aquest perfil, ja que el % d'èxit és més elevat.

# Variable 'marital'
summary(dd$marital) # Hi ha 80 persones de les quals no sabem el seu estat civil.
bar5 <- ggplot(dd, aes(x = marital, fill = y)) + 
        geom_bar(color = 'black') + 
        scale_fill_brewer(palette = 'Spectral') + 
        ggtitle('Depostits vs marital status') + 
        labs(x = 'Civil status', y = 'Number of people') + 
        theme(plot.title = element_text(hjust = 0.5)) +
        guides(fill = guide_legend(title = 'Success'))
bar5
marital <- table(dd$marital, dd$y)
marital / rowSums(marital)
# A priori no veiem cap diferència significativa, per tant sembla que l'estat civil no hauria d'influir.

# Variable 'education'
summary(dd$education) #Aquí tenim 1731 NA's, la qual cosa podria ser un problema
#EXPERIMENTS: com la variable education està ordenada, la codifiquem com a ordenada dins de R
dd$education <- factor(dd$education, levels = c('illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree'), ordered = TRUE)
class(dd$education)
# En aquest cas, la variable 'education' podem considerar que és una variable categòrica ordenada, ja que clarament
bar6 <- ggplot(dd, aes(x = education, fill = education)) +
        geom_bar(color = 'black') +
        scale_fill_brewer(palette = 'Spectral') +
        ggtitle('Education') +
        labs(x = 'Levels of education', y = 'Number of people') +
        theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90, hjust = 1), legend.position = 'none')
bar6 #Sembla que el major nombre de persones té nivell universitari o nivell de 'batxillerat'
bar7 <- ggplot(dd, aes(x = education, fill = y)) + 
        geom_bar(color = 'black') + 
        scale_fill_brewer(palette = 'Spectral') + 
        ggtitle('Deposits vs Education') + 
        labs(x = 'Levels of education', y = 'Number of people') + 
        theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90, hjust = 1)) +
        guides(fill = guide_legend(title = 'Success'))
bar7
# Sembla que a partir d'aquest gràfic no podem extreure conclusions, anem a comparar els percentatges de gent que sí fa un dipòsit depenent de la seva feina.
edu <- table(dd$education, dd$y)
edu
edu.freq <- edu / rowSums(edu)
edu.freq
# En un principi sembla que el percentatge d'èxit en grups com els 'retired' i els 'student' és més alt. Anem a comprobar-ho gràficament.
edu.freq.plot <- data.frame(edu.freq[,2]) #ggplot requires a dataframe to plot something
names(edu.freq.plot) <- c('success')
bar8 <- ggplot(edu.freq.plot, aes(x = rownames(edu.freq.plot), y = success, fill = rownames(edu.freq.plot))) +
        geom_bar(stat = 'identity', color = 'black') +
        scale_fill_brewer(palette = 'Spectral') +
        ggtitle('% of campaign success per education level') +
        labs(x = 'Levels of education', y = '% of yes') +
        theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90, hjust = 1), legend.position = 'none')
bar8
# Sembla ser que potser els clients amb estudis universitaris tenen un percentatge més alt. També els 'illiterate', que tindria sentit, però al tenir una representació tant baixa tampoc en podem extreure conclusions definitives.

# Variable 'default'
summary(dd$default) #Aquí tenim moltíssimes dades mancants, i només 3 de positives.
def <- which(dd$default == 'yes') #people who failed to repay a loan
dd[def,ncol(dd)] #sembla que evidentment les persones que tenien default no contracten un dipòsit, però a part d'això no podem dir res més a priori

# Variable 'housing'
summary(dd$housing) #tenim 990 NA's
houses <- table(dd$housing, dd$y)
houses
houses.freq <- houses / rowSums(houses)
houses.freq
# Aparentment no hi ha diferències significatives pel que fa a la variable 'housing'.

# Variable 'loan'
summary(dd$loan) #també tenim 990 NA's
# Anem a comprovar si són els mateixos NA's que la variable anterior (ja que les dues variables haurien d'estar relacionades)
all.equal(which(is.na(dd$housing)), which(is.na(dd$loan))) #efectivament són les mateixes així que les podríem tractar de la mateixa manera
loans <- table(dd$loan, dd$y)
loans
loans.freq <- loans / rowSums(loans)
loans.freq
# Aparentment tampoc hi ha diferències significatives pel que fa a la variable 'loan'

###
# Les edats no signifiquen massa, tenen una dispersió mitjana i no té gaire sentit relacionar-la amb les altres variables, ja que sembla que no estan relacionades.
# Les variables jobs, marital i education sembla que no tenen relació entre elles ni amb loan housing ni default, ja que si les creuem no tenen relació.
###

# Variable 'contact'
summary(dd$contact)
contacts <- table(dd$contact, dd$y)
contacts
contacts.freq <- contacts / rowSums(contacts)
contacts.freq #aquí sí que sembla que tenim una certa influència pel que fa al mètode de les trucades. Veiem que el % d'èxit de les trucades fetes a telèfons fixos és gairebé el triple que el realitzat a telèfons mòbils.

# Variable 'month'
summary(dd$month) #falten Gener i Febrer
bar9 <- ggplot(dd, aes(x = month, fill = month)) +
        geom_bar(color = 'black') +
        scale_fill_brewer(palette = 'Spectral') +
        ggtitle('Campaign distribution per months') +
        labs(x = 'Month', y = 'Number of people') +
        theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90, hjust = 1), legend.position = 'none')
bar9 #observem que la majoria de trucades es fan als mesos d'agost, juliol, juny, maig i novembre
bar10 <- ggplot(dd, aes(x = month, fill = y)) +
         geom_bar(color = 'black') +
         scale_fill_brewer(palette = 'Spectral') +
         ggtitle('Campaign success per month') +
         labs(x = 'Month', y = 'Number of people') +
         theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90, hjust = 1)) +
         guides(fill = guide_legend(title = 'Success'))
bar10
mon <- table(dd$month, dd$y)
mon
mon.freq <- mon / rowSums(mon)
mon.freq 
# Aparentment hi ha una variabilitat percentual bastant gran depenent del mes en el que es fan les trucades per la campanya de marketing, anem a visualitzar-ho.
mon.freq.plot <- data.frame(mon.freq[,2])
names(mon.freq.plot) <- c('success')
bar11 <- ggplot(mon.freq.plot, aes(x = rownames(mon.freq.plot), y = success, fill = rownames(mon.freq.plot))) +
         geom_bar(stat = 'identity', color = 'black') +
         scale_fill_brewer(palette = 'Spectral') +
         ggtitle('% of campaign success per month') +
         labs(x = 'Month', y = '% of yes') +
         theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90, hjust = 1), legend.position = 'none')
bar11
# A priori sembla ser que les trucades realitzades al desembre, març, octubre i septembre tenen un percentatge d'èxit més alt que les altres.

# Variable 'day_of_week'
summary(dd$day_of_week) #obserem que les trucades només es fan en dies laborables, i no en dies festius (cap de setmana).
table(dd$day_of_week) / N #sembla ser que les trucades estan equidistribuïdes també al llarg dels dies de la setmana.
days <- table(dd$day_of_week, dd$y)
days
days.freq <- days / rowSums(days)
days.freq
# No sembla que hi hagi diferències molt substancials pel que fa al percentatge d'èxit de les trucades en funció del dia de la setmana a priori.

# Variable 'duration'
# Temps de durada en segons de les trucades que formen part de la campanya
summary(dd$duration)
boxplot_duration <- ggplot(dd, aes(y, duration)) +
                    geom_boxplot(aes(fill = y)) +
                    ggtitle('Deposits vs Call duration') + 
                    labs(x = 'Success?', y = 'Call duration (s)') + 
                    theme(plot.title = element_text(hjust = 0.5), legend.position = 'none')
boxplot_duration
# Com veiem en el boxplot aquesta variable afecta bastant a la nostra variable resposta.
# Un exemple fàcil serien els següents casos
dur0 <- which(dd$duration == 0)
dd[dur0, ncol(dd)] #evidentment, quan la duració de la trucada és 0, el resultat és que el client no contractarà un dipòsit.
# Des del punt de vista d'un banc, la duració de la trucada no se sap abans de realitzar la trucada, per tant no ens pot ajudar si el que volem és un model predictiu real.
# Com a molt la podem utilitzar per motius de 'benchmark', que no és el nostre objectiu aquí.
# Per tant treiem aquesta variable del nostre dataset
dd <- dd[,-which(names(dd) == 'duration')]

# Variable 'campaign'
summary(dd$campaign) #és el nombre de vegades que s'ha trucat a aquest mateix client en la mateixa campanya de marketing.
boxplot(dd$campaign) #podem veure que la majoria de vegades que es truca a un mateix client en una campanya està entre 1 i 10, però en el boxplot podem veure una 'cua' molt llarga, la qual cosa ens indica que per alguna raó a alguns clients els han trucat moltíssimes vegades.
boxplot_campaign <- ggplot(dd, aes(y, campaign)) +
                    geom_boxplot(aes(fill = y)) +
                    ggtitle('Deposits vs #Campaign Calls') +
                    labs(x = 'Success?', y = '#Campaign Calls') +
                    theme(plot.title = element_text(hjust = 0.5), legend.position = 'none')
boxplot_campaign


# La variable 'pdays' representa el nombre de dies que han passat des de l'últim contacte de la companyia. Els nombres marcats amb 999 representen aquells que mai han estat contactats.
length(which(dd$pdays == 999)) #39673 mai havien estat contactats prèviament (aparentment)
length(which(dd$previous == 0)) #35563
length(which(dd$poutcome == 'nonexistent')) #35563
all.equal(which(dd$previous == 0), which(dd$poutcome == 'nonexistent')) #TRUE
# Per tant veiem com els números de previous i poutcome coincideixen i són els mateixos, i és que aquests clients mai havíen estat contactats prèviament en cap altra campanya de marketing.
# Però veiem com en pdays, tenim un número de 999 més gran, per tant podria ser que hi ha gent que va estar contactada fa més de 999 dies, és a dir fa més d'uns 2.7 anys.
all(which(dd$previous == 0) %in% which(dd$pdays == 999)) #TRUE
# Anem a detectar quins són aquells clients codificats com a 999 però que sí que havíen estat contactats prèviament.
which(dd$pdays == 999 & dd$previous != 0) #Conjunt de gent que van estar contactats en altres campanyes però fa molt temps!!!!

# COM ELS CODIFIQUEM PER DIFERENCIAR-LOS!?!?!?!?!?!?

# Per fer una visualització prèvia de la influència de la variable pdays, treurem els valors que són igual a 999
summary(dd$pdays[which(dd$pdays != 999)])
boxplot_pdays <- ggplot(dd[-which(dd$pdays == 999),], aes(y, pdays)) +
                 geom_boxplot(aes(fill = y)) +
                 ggtitle('Deposits vs pdays') +
                 labs(x = 'Success?', y = 'pdays') +
                 theme(plot.title = element_text(hjust = 0.5), legend.position = 'none')
boxplot_pdays

# Per tant decidim categoritzar la variable 'pdays' en 3 classes diferents
dd$pdays[which(dd$pdays < 999)] <- 'less.1month'
dd$pdays[which(dd$pdays == 999 & dd$previous != 0)] <- 'others'
dd$pdays[which(dd$pdays == 999)] <- 'never.contacted'
dd$pdays <- as.factor(dd$pdays)
summary(dd$pdays)

# Variable 'poutcome'
summary(dd$poutcome)
pout <- table(dd$poutcome, dd$y)
pout
pout.freq <- pout / rowSums(pout)
pout.freq
# Sembla ser que aquesta variable influeix molt en la nostra variable resposta.
pout.freq.plot <- data.frame(pout.freq[,2])
names(pout.freq.plot) <- c('success')
bar12 <- ggplot(pout.freq.plot, aes(x = rownames(pout.freq.plot), y = success, fill = rownames(pout.freq.plot))) +
         geom_bar(stat = 'identity', color = 'black') +
         scale_fill_brewer(palette = 'Spectral') +
         ggtitle('% of campaign success per poutcome') +
         labs(x = 'Outcome of last campaign', y = '% of yes') +
         theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 90, hjust = 1), legend.position = 'none')
bar12
# Evidentment, sembla que la probabilitat de convèncer a algú que mai havia estat contactat prèviament és bastant baixa, en canvi, aquells que ja havíen estat convençuts, són més propensos a tornar a caure en la campanya de marketing que els altres.


# social and economic context attributes
# - emp.var.rate: employment variation rate - quarterly indicator (numeric)
# - cons.price.idx: consumer price index - monthly indicator (numeric) 
# - cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
# - euribor3m: euribor 3 month rate - daily indicator (numeric)
# - nr.employed: number of employees - quarterly indicator (numeric)

# Variable 'emp.var.rate'
summary(dd$emp.var.rate)

# Variable 'cons.price.idx'
summary(dd$cons.price.idx)

# Variable 'cons.conf.idx'
summary(dd$cons.conf.idx)

# Variable 'euribor3m'
summary(dd$euribor3m)

# Variable 'nr.employed'
summary(dd$nr.employed)

# Mirem tots els boxplots juntament
par(mfrow = c(2, 3))
for (i in 15:19) {
  boxplot(dd[,i], main = names(dd)[i])
}
par(mfrow = c(2,3))
for (i in 15:19) {
  hist(dd[,i], col = 'light blue', main = names(dd)[i], xlab = '', ylab = '')
}
# Com veiem, la distribució de les nostres variables suggereixen una alta correlació entre algunes d'elles, anem a comprovar-ho més acuradament:
###
context <- as.matrix(dd[, c('emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed')])
context.cor <- cor(context)
context.cor
# Efectivament, veiem que moltes d'aquestes variables estan altament correlades, amb el qual podríem prescindir d'aquelles més correlades per predir el nostre model --> com menys variables explicatives, guanyem en interpretabilitat.
# Per exemple, les dues últimes variables (euribor3m, nr.employed) estan altament correlades amb la primera variable (emp.var.rate), concretament la correlació és més alta del 90%. Per tant, per modelar i simplificar, podríem perscindir d'elles.
# De totes maneres, provarem els nostres models amb elles i sense elles per veure com podem obtenir els millors resultats.
###


###
# Anem a analitzar com estan distribuïdes les dades mancants
library(mice)
library(VIM)

# Seleccionem només les variables que tenen NA's
dd.na <- dd[, c('default', 'education', 'housing', 'loan', 'job', 'marital')]


mice_plot <- aggr(dd.na, col = c('white', 'gold2'), numbers = TRUE, sortVars = TRUE, labels = names(dd.na), cex.axis = .7, gap = 3, ylab = c('Missing values', 'Combinations'))
# Com s'interpreta?
# L'histograma de l'esquerra representa en percentatges la quantitat de 'missing values' que té cada variable.
# El gràfic de la dreta en canvi, s'interpreta de la següent manera: hi ha un 74% de dades que no tenen cap missing value, un 19% que només tenen default com a missing value, un 2.7%que només tenen education com a missing value, etc.
###

###
# IMPUTING
###
library(missForest)
dd.imp <- missForest(dd, mtry = 10, ntree = 200, maxit = 10, verbose = TRUE)
dd.complete <- dd.imp$ximp
sum(is.na(dd.complete)) #ha omplert tots els NA
dd.imp$OOBerror[2] # Error derivat de l'imputing de les nostres variables categòriques. Proporció de variables mal classificades (estimació).



### Confirmatory Analyses


library(MASS)
library(DescTools)
library(xgboost)
library(Matrix)



### 1. ORMs of the same variables


## Older Generation

set.seed(1234567)

cm1.2 = polr(lonely ~ liveAlone * ipip20es + liveAlone*sex + widowed + self_rated_health + socialclass 
            , data=df.lbc[complete.cases(df.lbc[,c('lonely','liveAlone','ipip20es','widowed','self_rated_health','socialclass')]),])
cm1.2$n
summary(cm1.2)

summary(o.m1.2)

o.cm.0 = polr(lonely ~ 1,
                 data=df.lbc[complete.cases(df.lbc[,c('lonely','liveAlone','ipip20es','widowed','self_rated_health','socialclass')]),])

PseudoR2(cm1.2, which='Nagelkerke')



## Younger Generation

cm.2 = polr(as.ordered(lonely) ~ 1 + ipip50es + ipip50e, 
           data=df.elsa.young[complete.cases(df.elsa.young[,c('lonely','ipip50es','ipip50e')]),])
cm.2$n
summary(cm.2)

summary(y.m.2)

y.cm.0 = polr(as.ordered(lonely) ~ 1,
                  data=df.elsa.young[complete.cases(df.elsa.young[,c('lonely','ipip50es','ipip50e')]),])

PseudoR2(cm.2, which='Nagelkerke')



### 2. Confirmatory models 


## Older

vars = c('lonely','sex','socialclass','education','nochildren',
         'ipip20e','ipip20a','ipip20c','ipip20es','ipip20i',
         'g','single','widowed','divorced','self_rated_health',
         'liveAlone','livewFamily')

lbc.preds = df.lbc[,vars]
lbc.preds$lonely = as.integer(lbc.preds$lonely)
lbc.preds = lbc.preds[complete.cases(
  lbc.preds[,c('lonely','liveAlone','ipip20es','widowed','self_rated_health','socialclass','sex')]),]

# colnames(lbc.preds)
lbc.preds = cbind(lbc.preds[,c(1:16)],NA,lbc.preds[,c(17)],NA)
colnames(lbc.preds)[17] = 'livewChild'
colnames(lbc.preds)[18] = 'livewFamily'
colnames(lbc.preds)[19] = 'livewOther'


lbc.preds$xgb = predict(thirtysix.xgb, xgb.DMatrix(as.matrix(lbc.preds[,-1])))

lbc.preds$polr = predict(o.m1.2, lbc.preds)

lbc.preds$null = predict(o.cm.0, lbc.preds)

lbc.preds$new = predict(cm1.2, lbc.preds)

table(as.integer(lbc.preds$polr), useNA='ifany')
table(as.integer(lbc.preds$null), useNA='ifany')
table(lbc.preds$lonely, useNA='ifany')


lbc.preds$xgb.error = lbc.preds$lonely - (lbc.preds$xgb + 1)

lbc.preds$polr.error = lbc.preds$lonely - as.integer(lbc.preds$polr)

lbc.preds$null.error = lbc.preds$lonely - as.integer(lbc.preds$null)

lbc.preds$new.error = lbc.preds$lonely - as.integer(lbc.preds$new)

lbc.preds$norm = lbc.preds$lonely - mean(lbc.preds$lonely)


lbc.MSE.xgb = mean(lbc.preds$xgb.error^2)
lbc.MSE.polr = mean(lbc.preds$polr.error^2)
lbc.MSE.null = mean(lbc.preds$null.error^2)
lbc.MSE.new = mean(lbc.preds$new.error^2)

lbc.MSE.null
lbc.MSE.xgb
lbc.MSE.polr
lbc.MSE.new



## Younger

vars = c('lonely','sex','simd16','HighQual',
  'NoChildren','ipip50e','ipip50a','ipip50c','ipip50es','ipip50i',
  'g','single','widowed','divorced','SubjectiveHealth',
  'liveAlone','livewFamily','livewOther'
)

elsa.preds = df.elsa.young[,vars]
elsa.preds$lonely = as.numeric(elsa.preds$lonely)
elsa.preds = elsa.preds[complete.cases(
  elsa.preds[,c('lonely','ipip50es','ipip50a','ipip50e','SubjectiveHealth')]),]


elsa.preds$xgb = predict(hagis.xgb, xgb.DMatrix(as.matrix(elsa.preds[,-1])))

elsa.preds$polr = predict(y.m.2, elsa.preds)

elsa.preds$null = predict(y.cm.0, elsa.preds)

elsa.preds$new = predict(cm.2, elsa.preds)


table(as.integer(elsa.preds$polr), useNA='ifany')
table(as.integer(elsa.preds$null), useNA='ifany')
table(elsa.preds$lonely, useNA='ifany')


## Scale ELSA predictions to have the same max as LBC
elsa.preds$lonely = elsa.preds$lonely * 5/3

elsa.preds$xgb = (elsa.preds$xgb + 1) * 5/3
elsa.preds$polr = as.numeric(as.character(elsa.preds$polr)) * 5/3
elsa.preds$null = as.numeric(as.character(elsa.preds$null)) * 5/3
elsa.preds$new = as.numeric(as.character(elsa.preds$new)) * 5/3


elsa.preds$xgb.error = elsa.preds$lonely - elsa.preds$xgb

elsa.preds$polr.error = elsa.preds$lonely - elsa.preds$polr

elsa.preds$null.error = elsa.preds$lonely - elsa.preds$null

elsa.preds$new.error = elsa.preds$lonely - elsa.preds$new

elsa.preds$norm = elsa.preds$lonely - mean(elsa.preds$lonely)


elsa.MSE.xgb = mean(elsa.preds$xgb.error^2)
elsa.MSE.polr = mean(elsa.preds$polr.error^2)
elsa.MSE.null = mean(elsa.preds$null.error^2)
elsa.MSE.new = mean(elsa.preds$new.error^2)

elsa.MSE.null
elsa.MSE.xgb
elsa.MSE.polr
elsa.MSE.new

### Younger generation training analysis

## N.B. Looking at interactions and deeper details of the XGBMs requires an
## external program "xgbfi" built on the same foundations as "xgboost".

library(xgboost)
library(OpenMPController)
library(Matrix)
library(DiagrammeR)
library(ggplot2)
library(MASS)
library(caret)
library(pdp)
library(DescTools)



### Restrict age range 

table(df.hagis$AgeBands2, useNA='ifany') # 9 and above is 70 and older

df.hagis.young = df.hagis[df.hagis$AgeBands2<9,]


table(df.elsa$dhager, useNA='ifany')

df.elsa.young = df.elsa[df.elsa$dhager<70,]



### xgboost

vars = c(
  'sex','simd16','HighQual',
  'NoChildren',
  'ipip50e','ipip50a','ipip50c','ipip50es','ipip50i',
  'g',
  'single','widowed','divorced','SubjectiveHealth',
  'liveAlone','livewFamily','livewOther'
  )
mat.hagis = as.matrix(df.hagis.young[!is.na(df.hagis.young$lonely),vars])
response = df.hagis.young$lonely[!is.na(df.hagis.young$lonely)]


### What objective (link) to use?
mean(as.integer(response)-1, na.rm=TRUE) # 0.168
var(as.integer(response), na.rm=TRUE) # 0.157
## similar - Poisson is reasonable.



param = list(booster='gbtree',mx_depth=7, objective='count:poisson', nthread=3, silent=0)
#hist(df.hagis.young$lonely)


set.seed(1234567)
start.time <- Sys.time()
hagis.gbm = train(x = mat.hagis,
                  y = as.integer(response)-1,
                  method = "xgbTree", #metric = "Rsquared",
                  trControl = trainControl(method = "cv", number = 10),
                  tuneLength = 10
                  , objective='count:poisson', nthread=3, booster='gbtree'
)

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken # Time difference of 1.862629 hours


print(hagis.gbm$bestTune)
max(hagis.gbm$results$Rsquared)
min(hagis.gbm$results$RMSE)
## R^2 and RMSE get the same parameters
hagis.gbm$results[hagis.gbm$results$Rsquared==max(hagis.gbm$results$Rsquared),]



### Best parameters

## Poisson 
set.seed(1234567)
#table(as.numeric(df.hagis$lonely[!is.na(df.hagis$lonely)])-1)
hagis.xgb = xgboost(param, label = as.numeric(df.hagis.young$lonely[!is.na(df.hagis.young$lonely)])-1,
                    data = mat.hagis, verbose=0,
                    nrounds=50, eta=0.3, max_depth=1, gamma=0, colsample_bytree=0.8,
                    min_child_weight=1, subsample=0.6666667
)
importance.hagis <- xgb.importance(feature_names=colnames(mat.hagis), model=hagis.xgb)
importance.hagis

g <- xgb.ggplot.importance(importance.hagis) 
g + theme_bw()



## Partial dependance plots

pdp.srh <- partial(hagis.gbm, pred.var = "SubjectiveHealth", plot = TRUE, rug = TRUE)
pdp.es <- partial(hagis.gbm, pred.var = "ipip50es", plot = TRUE, rug = TRUE)
pdp.g <- partial(hagis.gbm, pred.var = "g", plot = TRUE, rug = TRUE)
pdp.es.g <- partial(hagis.gbm, pred.var = c("ipip50es", "g"),
                        plot = TRUE, chull = TRUE)
grid.arrange(pdp.es, pdp.g, pdp.es.g, ncol = 3)



### Exporting for interaction checking
featureList <- colnames(mat.hagis)
featureVector <- c() 
for (i in 1:length(featureList)) { 
  featureVector[i] <- paste(i-1, featureList[i], "q", sep="\t") 
}
write.table(featureVector, "fmap.txt", row.names=FALSE, quote = FALSE, col.names = FALSE)
xgb.dump(model = hagis.xgb, fname = 'xgb.dump', fmap = "fmap.txt", with_stats = TRUE)



### Single variable ordinal model building

y.lm.0 = lm(lonely ~ 1, 
           data=df.hagis.young[complete.cases(df.hagis.young[,c('lonely','ipip50es')]),])
summary(y.lm.0)
plot(y.lm.0)

y.m.0 = polr(as.ordered(lonely) ~ 1, 
           data=df.hagis.young[complete.cases(df.hagis.young[,c('lonely','ipip50es')]),])
y.m.0$n


y.m.1 = polr(as.ordered(lonely) ~ 1 + ipip50es, 
           data=df.hagis.young[complete.cases(df.hagis.young[,c('lonely','ipip50es'
                                                    , 'ipip50e' # comment in and out for comparing models with same data size
                                                    )]),])
y.m.1$n
summary(y.m.1)

anova(y.m.0, y.m.1)


y.m.2 = polr(as.ordered(lonely) ~ 1 + ipip50es + ipip50e, 
           data=df.hagis.young[complete.cases(df.hagis.young[,c('lonely','ipip50es','ipip50e'
#                                                    , 'g' # comment in and out for comparing models with same data size
                                                    )]),])
y.m.2$n
summary(y.m.2)

anova(y.m.1, y.m.2)



y.m.3 = polr(as.ordered(lonely) ~ 1 + ipip50es + ipip50e + g
                      , data=df.hagis.young[complete.cases(df.hagis.young[,c('lonely','ipip50es','ipip50e','g'
                                                                 #, 'SubjectiveHealth' # comment in and out for comparing models with same data size
                      )]),])
y.m.3$n
summary(y.m.3)

anova(y.m.2, y.m.3)

### Stop at y.m.2 ###



### Inteaction depth 1
## no interactions in the spreadsheet for HAGIS



### R^2 type measures

PseudoR2(y.m.1, which='Nagelkerke')

PseudoR2(y.m.2, which='Nagelkerke')

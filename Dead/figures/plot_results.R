library(dplyr)
library(ggplot2)
library(tidyr)

df<-read.csv("/Users/benweinstein/Documents/DeepForest_Model/Dead/figures/results.csv")
df$Dead[df$Dead==0]<-"Alive"
df$Dead[df$Dead==1]<-"Dead"

df %>% group_by(plantStatus, Dead) %>% summarize(n=n()) %>% pivot_wider(names_from = Dead, values_from=n) %>% mutate(recall=Dead/(Alive+Dead)*100)

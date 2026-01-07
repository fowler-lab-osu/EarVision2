library(ggplot2)
library(dplyr)

A <- read.csv("A_InferenceOutput-Jose_Aug8-24.csv")
B <- read.csv("B_InferenceOutput-Jose_Aug8-24.csv")
X <- read.csv("X_InferenceOutput-Jose_Aug8-24.csv")
Y <- read.csv("Y_InferenceOutput-Jose_Aug8-24.csv")
Z <- read.csv("Z_InferenceOutput-Jose_Aug8-24.csv")


A <- A %>% filter(!(CrossType=="Other" ) & is.na(TrainingSet))
B <- B %>% filter(!(CrossType=="Other" ) & is.na(TrainingSet))
X <- X %>% filter(!(CrossType=="Other" ) & is.na(TrainingSet))
Y <- Y %>% filter(!(CrossType=="Other" ) & is.na(TrainingSet))
Z <- Z %>% filter(!(CrossType=="Other" ) & is.na(TrainingSet))

all <- rbind(A,B,X,Y,Z)

A["Passes"] <- A["AmbiguousKernelPercentage"]<=2 & A["PredictedTotal"]>100
B["Passes"] <- B["AmbiguousKernelPercentage"]<=2 & B["PredictedTotal"]>100
X["Passes"] <- X["AmbiguousKernelPercentage"]<=2 & X["PredictedTotal"]>100
Y["Passes"] <- Y["AmbiguousKernelPercentage"]<=2 & Y["PredictedTotal"]>100
Z["Passes"] <- Z["AmbiguousKernelPercentage"]<=2 & Z["PredictedTotal"]>100
all["Passes"] <- all["AmbiguousKernelPercentage"]<=2 & all["PredictedTotal"]>100


A_filtered <- A %>% filter(Passes==TRUE)
B_filtered <- B %>% filter(Passes==TRUE)
X_filtered <- X %>% filter(Passes==TRUE)
Y_filtered <- Y %>% filter(Passes==TRUE)
Z_filtered <- Z %>% filter(Passes==TRUE)
all_filtered <- all %>% filter(Passes==TRUE)


A_total <- nrow(A)
A_passes <- nrow(A_filtered)
A_flagged <- A_total-A_passes
#find passing and flagged for each year that also have entires for 'actual transmission' and get plotted
A_passes_plotted <- sum(!is.na(A_filtered["ActualTransmission"]))
A_flagged_plotted <- sum(!is.na(A["ActualTransmission"])) - A_passes_plotted 

B_total <- nrow(B)
B_passes <- nrow(B_filtered)
B_flagged <- B_total - B_passes
B_passes_plotted <- sum(!is.na(B_filtered["ActualTransmission"]))
B_flagged_plotted <- sum(!is.na(B["ActualTransmission"])) - B_passes_plotted 

X_total <- nrow(X)
X_passes <- nrow(X_filtered)
X_flagged <- X_total-X_passes
X_passes_plotted <- sum(!is.na(X_filtered["ActualTransmission"]))
X_flagged_plotted <- sum(!is.na(X["ActualTransmission"])) - X_passes_plotted 


Y_total <- nrow(Y)
Y_passes <- nrow(Y_filtered)
Y_flagged <- Y_total-Y_passes
Y_passes_plotted <- sum(!is.na(Y_filtered["ActualTransmission"]))
Y_flagged_plotted <- sum(!is.na(Y["ActualTransmission"])) - Y_passes_plotted 


Z_total <- nrow(Z)
Z_passes <- nrow(Z_filtered)
Z_flagged <- Z_total-Z_passes
Z_passes_plotted <- sum(!is.na(Z_filtered["ActualTransmission"]))
Z_flagged_plotted <- sum(!is.na(Z["ActualTransmission"])) - Z_passes_plotted 


all_total <- nrow(all)
all_passes <- nrow(all_filtered)
all_flagged <- all_total - all_passes
all_passes_plotted <- sum(!is.na(all_filtered["ActualTransmission"]))
all_flagged_plotted <- sum(!is.na(all["ActualTransmission"])) - all_passes_plotted 


total_ears <- c(A_total, B_total,Z_total,Y_total, X_total, all_total)
passed <- c(A_passes, B_passes, Z_passes, Y_passes, X_passes, all_passes)
flagged <- c(A_flagged, B_flagged, Z_flagged, Y_flagged, X_flagged, all_flagged)
proportion_flagged <- c(A_flagged/A_total, B_flagged/B_total, Z_flagged/Z_total, Y_flagged/Y_total, X_flagged/X_total, all_flagged/all_total)
percentage_flagged <- c(A_flagged/A_total*100.0, B_flagged/B_total*100.0, Z_flagged/Z_total*100.0, Y_flagged/Y_total*100.0, X_flagged/X_total*100.0, all_flagged/all_total*100.0)
plotted_passed <- c(A_passes_plotted, B_passes_plotted,Z_passes_plotted,Y_passes_plotted, X_passes_plotted, all_passes_plotted)
plotted_flagged <-c(A_flagged_plotted, B_flagged_plotted,Z_flagged_plotted,Y_flagged_plotted, X_flagged_plotted, all_flagged_plotted)

counts <- data.frame(year=c("A","B","Z","Y","X","all"),total_ears, passed, flagged, proportion_flagged, percentage_flagged, plotted_passed, plotted_flagged)

print(counts)

write.csv(counts, "ear_flagged_counts.csv")


correlation_values = data.frame()
correlation_values["A","Inclusive"] <- cor(A$ActualTransmission, A$PredictedTransmission, use="complete.obs", method="pearson")
correlation_values["A","Filtered"] <- cor(A_filtered$ActualTransmission, A_filtered$PredictedTransmission, use="complete.obs", method="pearson")

correlation_values["B","Inclusive"] <- cor(B$ActualTransmission, B$PredictedTransmission, use="complete.obs", method="pearson")
correlation_values["B","Filtered"] <- cor(B_filtered$ActualTransmission, B_filtered$PredictedTransmission, use="complete.obs", method="pearson")

correlation_values["Z","Inclusive"] <- cor(Z$ActualTransmission, Z$PredictedTransmission, use="complete.obs", method="pearson")
correlation_values["Z","Filtered"] <- cor(Z_filtered$ActualTransmission, Z_filtered$PredictedTransmission, use="complete.obs", method="pearson")

correlation_values["Y","Inclusive"] <- cor(Y$ActualTransmission, Y$PredictedTransmission, use="complete.obs", method="pearson")
correlation_values["Y","Filtered"] <- cor(Y_filtered$ActualTransmission, Y_filtered$PredictedTransmission, use="complete.obs", method="pearson")


correlation_values["X","Inclusive"] <- cor(X$ActualTransmission, X$PredictedTransmission, use="complete.obs", method="pearson")
correlation_values["X","Filtered"] <- cor(X_filtered$ActualTransmission, X_filtered$PredictedTransmission, use="complete.obs", method="pearson")


correlation_values["all","Inclusive"] <- cor(all$ActualTransmission, all$PredictedTransmission, use="complete.obs", method="pearson")
correlation_values["all","Filtered"] <- cor(all_filtered$ActualTransmission, all_filtered$PredictedTransmission, use="complete.obs", method="pearson")


correlation_values["Inclusive_R2"] <- correlation_values["Inclusive"]*correlation_values["Inclusive"]
correlation_values["Filtered_R2"] <- correlation_values["Filtered"]*correlation_values["Filtered"]


print(correlation_values)
write.csv(correlation_values, "correlation_values.csv")

#color_pair_p<- c("#3c8ba5BB", "#AA040a66")
#color_pair_p_outline<- c("#3c8ba5e7", "#aa040aaa")

#invisible non-flagged:
#color_pair_p<- c("#90030866","#78a3b200")
#color_pair_p_outline<- c("#990207aa","#00000000")


color_pair_p<- c("#990207aa","#43a1b7d4")



the_theme <- theme(panel.background = element_rect(fill = '#f1f1f1'), 
plot.margin = margin(5,10,5,10) , legend.position="none", plot.title = element_text(hjust = 0.5, size=23),
axis.title = element_text(size=28), axis.text = element_text(size = 20) )

point_size <- 2

w = 7
h = 6
res = 300


ggplot(A, aes(x = ActualTransmission, y = PredictedTransmission)) + 
geom_point(size=point_size, aes(color = Passes)) + xlab("ImageJ Annotation") +
ylab("EarVision.v2 Prediction")+ scale_color_manual(values=color_pair_p) + the_theme +
geom_smooth(data=A_filtered, aes(x = ActualTransmission, y = PredictedTransmission), method="lm", color="#00000066", fullrange=TRUE) +
scale_fill_manual(values=color_pair_p) + ggtitle("2021")+xlim(0,100)
ggsave("A_2021.pdf", width=w, height=h)
ggsave("A_2021.png", width=w, height=h, dpi=res)

ggplot(B, aes(x = ActualTransmission, y = PredictedTransmission)) +
geom_point(size=point_size, aes(color = Passes)) + xlab("ImageJ Annotation") +
ylab("EarVision.v2 Prediction")+ scale_color_manual(values=color_pair_p) + the_theme +
geom_smooth(data=B_filtered, aes(x = ActualTransmission, y = PredictedTransmission), method="lm", color="#00000066", fullrange=TRUE)+ 
scale_fill_manual(values=color_pair_p) + ggtitle("2022")+xlim(0,100)
ggsave("B_2022.pdf", width=w, height=h)
ggsave("B_2022.png", width=w, height=h, dpi=res)

ggplot(X, aes(x = ActualTransmission, y = PredictedTransmission)) +
geom_point(size=point_size, aes(color = Passes)) + xlab("ImageJ Annotation") +
ylab("EarVision.v2 Prediction")+ scale_color_manual(values=color_pair_p) + the_theme +
geom_smooth(data=X_filtered, aes(x = ActualTransmission, y = PredictedTransmission), method="lm", color="#00000066", fullrange=TRUE) +
scale_fill_manual(values=color_pair_p) + ggtitle("2018") +xlim(0,100)
ggsave("X_2018.pdf", width=w, height=h)
ggsave("X_2018.png", width=w, height=h, dpi=res)

ggplot(Y, aes(x = ActualTransmission, y = PredictedTransmission)) +
geom_point(size=point_size, aes(color = Passes)) + xlab("ImageJ Annotation") +
ylab("EarVision.v2 Prediction")+ scale_color_manual(values=color_pair_p) + the_theme +
geom_smooth(data=Y_filtered, aes(x = ActualTransmission, y = PredictedTransmission), method="lm", color="#00000066", fullrange=TRUE) +
scale_fill_manual(values=color_pair_p) + ggtitle("2019")+xlim(0,100)
ggsave("Y_2019.pdf", width=w, height=h)
ggsave("Y_2019.png", width=w, height=h, dpi=res)

ggplot(Z, aes(x = ActualTransmission, y = PredictedTransmission)) +
geom_point(size=point_size, aes(color = Passes)) + xlab("ImageJ Annotation") +
ylab("EarVision.v2 Prediction")+ scale_color_manual(values=color_pair_p) + the_theme +
geom_smooth(data=Z_filtered, aes(x = ActualTransmission, y = PredictedTransmission), method="lm", color="#00000066", fullrange=TRUE) +
scale_fill_manual(values=color_pair_p) + ggtitle("2020")+xlim(0,100)
ggsave("Z_2020.pdf", width=w, height=h)
ggsave("Z_2020.png", width=w, height=h, dpi=res)

ggplot(all, aes(x = ActualTransmission, y = PredictedTransmission)) +
geom_point(size=point_size, aes(col = Passes)) + xlab("ImageJ Annotation") +
ylab("EarVision.v2 Prediction")+  scale_color_manual(values=color_pair_p) + the_theme +
geom_smooth(data=all_filtered, aes(x = ActualTransmission, y = PredictedTransmission), method="lm", color="#00000066", fullrange=TRUE) 


ggsave("all_years_performance.pdf", width=w, height=h)
ggsave("all_years_performance.png", width=w, height=h, dpi=res)
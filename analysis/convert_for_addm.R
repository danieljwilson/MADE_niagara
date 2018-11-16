
library(tidyverse)

load("swap_v1_clean.RData")
load("swap_v2_clean.RData")

# face coded as
## 0 [2]
# house coded as 
## 1 [1]


df_input = swap_v2_clean
df = NULL # output

# v2 has some really high number of swap trials
max(df_input$swapAmount)
# Getting rid of all above 15
df$`16_fixation` <- NA    

# get rid of choice NAs
df_input = df_input[complete.cases(df_input[ , 1]),]

# find the first NA val in each row
naVal <- vector(mode="numeric", length=0)
for(x in 1:length(df_input$trial)){
  naVal[x] <- min(which(is.na(df_input[x,])))
}

# Check that we are starting at column fix#1
firstFix = which(names(df_input[1,]) == "1_fixation")

# reset row numbers
rownames(df_input) <- NULL

              
i = 1
for(x in 1:length(df_input$trial)){
  j = 1
  for(y in firstFix:(naVal[x]-1)){
    df$subject[i] = df_input$subject[x]
    df$earnings[i] = df_input$final_earnings[x]
    df$trial[i] = df_input$trial[x]
    df$faceVal[i] = df_input$face_val_base[x] 
    df$houseVal[i] = df_input$house_val_base[x]
    df$multFace[i] = df_input$face_mult[x]
    df$multHouse[i] = df_input$house_mult[x]
    df$totValFace[i] = df_input$face_val_total[x]
    df$totValHouse[i] = df_input$house_val_total[x]
    df$summedVal[i] = df_input$summed_val[x]
    
    df$rt[i] = df_input$rt[x]
    df$choice[i] = df_input$choice[x]
    df$correct[i] = df_input$correct[x]
    
    # for ROI 0 = FACE, 1 = HOUSE
    df$roi[i] = (df_input$first_image[x] + j - 1) %% 2   # Goes between ROI 0 and 1 with each Fixation
    df$fixNum[i] = j  
    df$revFixNum[i] =  naVal[x] - firstFix -j +1 
    
    df$fixDur[i] = df_input[x,y]

    df$firstFix[i] = df_input$first_image[x]
    df$finalFix[i] = df_input$last_image[x]
    j = j+1
    i = i+1
  }
}

df_K = data.frame("subject" = df$subject, 
                  "earnings" = df$earnings, 
                  "trial" = df$trial, 
                  "faceVal" = df$faceVal, 
                  "houseVal" = df$houseVal, 
                  "multFace" = df$multFace,
                  "multHouse" = df$multHouse, 
                  "totValFace" = df$totValFace, 
                  "totValHouse" = df$totValHouse, 
                  "summedVal" = df$summedVal, 
                  "rt" = df$rt, 
                  "choice" = df$choice, 
                  "correct" = df$correct,
                  "roi" = df$roi, 
                  "fixNum" = df$fixNum, 
                  "revFixNum" = df$revFixNum, 
                  "fixDur" = df$fixDur, 
                  "firstFix" = df$firstFix, 
                  "finalFix" = df$finalFix)

fixation_intermediary_df <- df_K

# BIN VALUES
library(dplyr)
df_input$face_bin = dplyr::ntile(df_input$face_val_total, 20)  # then remove the bottom and top (5% on either end, a la ratcliff)
df_input$house_bin = dplyr::ntile(df_input$house_val_total, 20)

# make bottom 5% bin 0 and top 5% bin 99
df_input$face_bin = plyr::mapvalues(df_input$face_bin, c(1,20), c(0,99))
df_input$house_bin = plyr::mapvalues(df_input$house_bin, c(1,20), c(0,99))

# remap 2/3->1, 4/5->2, etc.
df_input$face_bin = plyr::mapvalues(df_input$face_bin, c(2:19), rep(c(1:9), each=2))
df_input$house_bin = plyr::mapvalues(df_input$house_bin, c(2:19), rep(c(1:9), each=2))

# find the center (median) of each bin
for (bin in 1:9){
  x[bin] = print(median(df_input$face_val_total[df_input$face_bin==bin]))
}
for (bin in 1:9){
  y[bin] = print(median(df_input$house_val_total[df_input$house_bin==bin]))
}

# fudge to make house and face the same (sum/2)
centered_values = rowMeans(cbind(x, y))
# create bin value columns
df_input$face_bin_c = plyr::mapvalues(df_input$face_bin, c(1:9), centered_values)
df_input$house_bin_c = plyr::mapvalues(df_input$house_bin, c(1:9), centered_values)

# MAKE choice -1 and 1 instead of 0 and 1 (reject/accept)
df_input$choice[df_input$choice==0] = -1

#---------------------------#
# Response DF               #
#---------------------------#

# parcode,trial,rt,choice,item_left,item_right,valid (don't actually need valid)

expdata <- data.frame("parcode" = df_input$subject , "trial" = df_input$trial,
                      "rt" = df_input$rt * 1000, # convert to ms 
                      "choice" = df_input$choice,
                      "face_val_tot" = df_input$face_val_total, "house_val_tot" = df_input$house_val_total,
                      "face_mult" = df_input$face_mult, "house_mult" = df_input$house_mult,
                      "face_bin" = df_input$face_bin, "house_bin" = df_input$house_bin,
                      "face_bin_val" = df_input$face_bin_c, "house_bin_val" = df_input$house_bin_c)

write.csv(expdata, file = "/Data/expdata_v2.csv", row.names=FALSE)


# from  fixation_intermediary_df we need:
# parcode, trial, fix_item, fix_time

fixations <- data.frame("parcode" = fixation_intermediary_df$subject,
                        "trial" = fixation_intermediary_df$trial,
                        "fix_item" = fixation_intermediary_df$roi + 1, # tavares uses 1 and 2 for left and right fixes
                        "fix_time" = fixation_intermediary_df$fixDur *1000, # convert to ms
                        "fix_num" = fixation_intermediary_df$fixNum,
                        "rev_fix_num" = fixation_intermediary_df$revFixNum)

write.csv(fixations, file = "/Users/djw/Dropbox/PROGRAMMING/*NEURO/aDDM_Tavares/made_data/fixations.csv", row.names=FALSE)




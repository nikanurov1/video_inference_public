import os
import torch
import pandas as pd




class TrackerResults:
    def __init__ (self, dir_expiriment,
                        tracker_columns,
                        save_best):
        self.dir_expiriment = dir_expiriment
        self.best_name = os.path.join(dir_expiriment, "best.pth")
        self.pth_csv_expiriment = os.path.join(dir_expiriment,'result.csv')
        
        self.__df_res = pd.DataFrame(columns = tracker_columns)
        self.__df_res.set_index('epoch')
        
        self.save_best=save_best
        

    def add_reslutsONEepoch(self, **data):
        
        # getting current row as Dataframe 
        last_row=self.fill_curret_row(data)

        # Concat (append) row (metric results) to Data-Frame
        self.__df_res = pd.concat([self.__df_res, last_row]) #, ignore_index=True
        self.__df_res.to_csv(path_or_buf=self.pth_csv_expiriment, sep=',')
         
        
    def select_save_top_by(self, key_metric, model):
        """
        Function do:
        1) save model checkpoint if curent model in TOP of self.save_best
        2) delete not best checkpoints if not in TOP self.save_best
        3) save best model over all
        """
        # Select current (last) epoch and  current (last) key_metris
        current_row=self.__df_res.iloc[-1]
        current_metric=current_row[key_metric]
        current_checkpoint=current_row["name_checkpoint"]
        
        ############ Save model if it in best of n (self.save_best) ############ 
        # if key_metric=='valid_acc' => need resort dataframe as ascending=False
        list_checkpoints=self.__df_res.sort_values(key_metric)["name_checkpoint"].tolist()
        if current_checkpoint in list_checkpoints[:self.save_best]:
            torch.save(model.state_dict(), os.path.join(self.dir_expiriment,current_checkpoint))
            
        ############ Delete not best checkpoints ############
        if len(list_checkpoints)>self.save_best:
            for del_check in list_checkpoints[self.save_best:]:
                del_path = os.path.join(self.dir_expiriment,del_check)
                if os.path.exists(del_path):
                    os.remove(del_path)
                    print(f"The file {del_check} has been deleted successfully !")
                    
        ############ save best model ############
        if current_metric<=min(self.__df_res[key_metric]):
            torch.save(model.state_dict(), self.best_name)
            print(f"Best models saved on epoch: {current_row['epoch']}, valid_loss: {current_metric}")
            
            
    @staticmethod   
    def fill_curret_row(data):
        """
        Function do:
        fill row as (py dict) to Dataframe for appedning (concat)
        """
        row={}
        for key, value in data.items():
            if key!="lr":
                row[key]=[round(value,4)]
            elif key=="lr":
                row[key]=[value]
        
        row["name_checkpoint"]=f"checkpoint_{str(data['epoch']).zfill(4)}.pth"
        last_row=pd.DataFrame(row)
        last_row.set_index('epoch')
        return last_row
        

#     @property
#     def empty_row(self):
#         tmp_dict={}
#         for key in self.tracker_columns:
#             tmp_dict[key]=[]
#         return tmp_dict
    
    @property
    def get_df(self):
        return self.__df_res.copy(deep=True)
    
    

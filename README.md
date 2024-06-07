Supposed there are six samples: 'aa', 'bb', 'bb', 'dd', 'ee', ff. And each of 'aa', 'bb', 'cc', 'dd' belongs to a certain category, on the contrary, 'ee' and 'ff' do not belong to any certain obvious category including class 'Alpha' or 'Beta'. 
Let's say 'aa', 'bb' belong to class 'Alpha', 'cc', 'ee' belong to class 'Beta'.
So what if we want to use deep learning model to classifiy these 'aa', 'bb', 'cc', 'dd' as good as possibile without 'ee', 'ff' being classified as either class 'Alpha' or 'Beta'?
If you have similar problem, please try this 'negative_penalty_cce_loss' proposed in this repo.

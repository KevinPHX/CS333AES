'''
We will be working with Essay Sets 1 & 2 from ASAP (both are persuasive prompts)

Relatively straightforward and feasible: 
1. Identify tokens and sentence boundaries -- segmenter 
2. Identify paragraphs by checking for line breaks -- check if these exist in ASAP dataset 
3. Lemmatize -- Mate Tools Lemmatizer 
4. POS tagger -- Stanford POS tagger 
5. Constituent and dependency parser -- Klein, Dan and Christopher D. Manning. 2003. Accurate unlexicalized parsing. In Proceedings of the 41st Annual Meeting of the Association for Computational Linguistics - Volume 1, ACL ’03, pages 423–430, Sapporo.
6. Sentiment analyzer -- Socher, Richard, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, WA.
8. DKPRO TC text classification framework for feature extraction and experimentation 

Problematic: 
7. Discourse parser -- Lin, Ziheng, Min-Yen Kan, and Hwee Tou Ng. 2009. Recognizing implicit discourse relations in the Penn Discourse Treebank. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing: Volume 1, EMNLP ’09, pages 343–351, Suntec.
'''
Replication study of a paper by Stab & Gurveych (2017) on the Hewlett's Foundation's Automated Student Assessment Prize (ASAP) dataset. 

- **Citation**: Stab, C., & Gurevych, I. (2017). Parsing Argumentation Structures in Persuasive Essays. *Computational Linguistics*, 43(3), 619–659. https://doi.org/10.1162/COLI_a_00295


## Datasets  
- Essay Set 1 (ASAP): Effects of computers on people 
- Essay Set 2 (ASAP): Censorship in libraries 
- Annotated Essay Dataset (Stab & Gurveych 2017): https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422

## Preprocessing 
**DKPro Core**: https://dkpro.github.io/dkpro-core/

- **Citation**: Eckart de Castilho, R. and Gurevych, I. (2014). A broad-coverage collection of portable NLP components for building shareable analysis pipelines. In *Proceedings of the Workshop on Open Infrastructures and Analysis Frameworks for HLT (OIAF4HLT) at COLING 2014*, p 1-11, Dublin, Ireland. 

### Pipeline
1. LanguageTool segmenter: https://dkpro.github.io/dkpro-core/releases/2.2.0/docs/component-reference.html#engine-LanguageToolSegmenter

    - **NOTE**: S&G identified paragraphs by checking for line breaks. There are no line breaks in the ASAP essays, so we will treat every essay as a single paragraph.    

2. MateTools Lemmatizer: https://dkpro.github.io/dkpro-core/releases/2.2.0/docs/component-reference.html#engine-MateLemmatizer

3. Stanford POS tagger: https://dkpro.github.io/dkpro-core/releases/2.2.0/docs/component-reference.html#engine-StanfordPosTagger

4. Stanford Constituent and Dependency Parsers: https://dkpro.github.io/dkpro-core/releases/2.2.0/docs/component-reference.html#engine-StanfordParser 

    - **NOTE**: Klein and Manning 2003 --> https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/parser/lexparser/LexicalizedParser.html 

5. Sentiment Analyzer: https://dkpro.github.io/dkpro-core/releases/2.2.0/docs/component-reference.html#engine-StanfordSentimentAnalyzer

    - **NOTE**: Unspecified, but the only available option in DKPro Core is the Stanford Sentiment Analyzer 

6. Discourse Parser

    - Original Ruby implementation by Lin et al.: https://github.com/linziheng/pdtb-parser
    - Java Implementation: https://github.com/WING-NUS/pdtb-parser
    - There is also the 'writePennTree' option for the Berkeley Parser and OpenNLP parser in the DKPro Core toolkit. 

## Feature Extraction 

DKPro TC text classification framework: https://dkpro.github.io/dkpro-tc/

**Citations**: 
- Johannes Daxenberger, Oliver Ferschke, Iryna Gurevych, and Torsten Zesch (2014). DKPro TC: A Java-based Framework for Supervised Learning Experiments on Textual Data. In: *Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (System Demonstrations)*, pp. 61-66, Baltimore, Maryland, USA. (pdf) (bib)
- Tobias Horsmann and Torsten Zesch (2018). DeepTC - An Extension of DKPro Text Classification for Fostering Reproducibility of Deep Learning Experiments. In: *Proceedings of the International Conference on Language Resources and Evaluation (LREC)*, pp. 2539-2545, Miyazaki, Japan. (pdf) (bib)
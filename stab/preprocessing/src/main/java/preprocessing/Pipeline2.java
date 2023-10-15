package preprocessing;

import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import org.ejml.simple.SimpleMatrix;

public class Pipeline2 {
	public static StanfordCoreNLP pipeline;
    
	public static void init() 
    {
        Properties props = new Properties();
		// default parser is PCFG 
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse,sentiment");
        pipeline = new StanfordCoreNLP(props);

    }

	public static Annotation annotate(String text){
		return pipeline.process(text);
	}
	public static void run_parser(Annotation annotation)
    {
		// sample code from the official documentation: https://stanfordnlp.github.io/CoreNLP/parse.html 
		// gets all the NP and VP constituents in each sentence 
    	for(CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class))
	    {
			Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
			System.out.println(tree);
			Set<Constituent> treeConstituents = tree.constituents(new LabeledScoredConstituentFactory());
			for (Constituent constituent : treeConstituents) {
				if (constituent.label() != null &&
					(constituent.label().toString().equals("VP") || constituent.label().toString().equals("NP"))) {
					System.err.println("found constituent: "+constituent.toString());
					System.err.println(tree.getLeaves().subList(constituent.start(), constituent.end()+1));
          }
        }
      } 
    }
	
    public static void run_sentiment_analysis(Annotation annotation, Boolean getDetails)
    {
    	int predictedScore;
	    String category; 
	    for(CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class))
	    {
			Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
			predictedScore = RNNCoreAnnotations.getPredictedClass(tree); 
			category = sentence.get(SentimentCoreAnnotations.SentimentClass.class);
			System.out.println(category + "\t" + predictedScore + "\t" + sentence);
			if(getDetails == true){
				SimpleMatrix fivescores = RNNCoreAnnotations.getPredictions(tree);
				double veryNegative = fivescores.get(0);
				double negative = fivescores.get(1);
				double neutral = fivescores.get(2);
				double positive = fivescores.get(3);
				double veryPositive = fivescores.get(4);
				System.out.println("\t" + "Very Negative: " + veryNegative);
				System.out.println("\t" + "Negative: " + negative);
				System.out.println("\t" + "Neutral: " + neutral);
				System.out.println("\t" + "Positive: " + positive);
				System.out.println("\t" + "Very Positive: " + veryPositive + "\n");
			}	
	    }
     }

    
    public static void token_sentiment(Annotation annotation)
    {
    	for (CoreLabel tok : annotation.get(CoreAnnotations.TokensAnnotation.class)) {
    		System.out.println(tok.word() + "\t" + tok.lemma() + "\t" + tok.tag());
    		String token = tok.word();
    		if (token.matches("^[a-zA-Z0-9]+")) {
    			Annotation token_annotation = Pipeline2.annotate(token);
            	Pipeline2.run_sentiment_analysis(token_annotation,false);
    		}
        }
    }

    
    public static void main(String[] args) 
    {
    	String sample = "What a fascinating book! I really loved reading it. Thank you for recommending it to me!";
    	Pipeline2.init();
		Annotation annotation = Pipeline2.annotate(sample);
		Pipeline2.run_parser(annotation);
    	Pipeline2.run_sentiment_analysis(annotation,true);
    	Pipeline2.token_sentiment(annotation);
    }
}
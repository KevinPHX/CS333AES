package dkpro_preprocessing;

import java.util.Properties;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import org.ejml.simple.SimpleMatrix;

public class SentimentAnalyzer {
	public static StanfordCoreNLP sentiment_pipeline, token_pipeline;
    
	public static void init() 
    {
        Properties props1 = new Properties();
        props1.setProperty("annotators", "tokenize,ssplit, parse, sentiment");
        sentiment_pipeline = new StanfordCoreNLP(props1);
        
        Properties props2 = new Properties();
        props2.setProperty("annotators", "tokenize");
        token_pipeline = new StanfordCoreNLP(props2);
    }
    public static void run_sentiment_analysis(String text)
    {
    	int predictedScore;
	    String category; 
	    Annotation annotation = sentiment_pipeline.process(text);
	    for(CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class))
	    {
	      System.out.println(sentence);
	      Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
	      predictedScore = RNNCoreAnnotations.getPredictedClass(tree); 
	      category = sentence.get(SentimentCoreAnnotations.SentimentClass.class);
	      SimpleMatrix fivescores = RNNCoreAnnotations.getPredictions(tree);
		  double veryNegative = fivescores.get(0);
		  double negative = fivescores.get(1);
		  double neutral = fivescores.get(2);
		  double positive = fivescores.get(3);
		  double veryPositive = fivescores.get(4);
		  System.out.println(category + "\t" + predictedScore + "\t" + sentence);
		  System.out.println("\t" + "Very Negative: " + veryNegative);
		  System.out.println("\t" + "Negative: " + negative);
		  System.out.println("\t" + "Neutral: " + neutral);
		  System.out.println("\t" + "Positive: " + positive);
		  System.out.println("\t" + "Very Positive: " + veryPositive + "\n");
	    }
     }
    public boolean isAlphaNumeric(String s){
        String pattern= "^[a-zA-Z0-9]*$";
        return s.matches(pattern);
    }
    
    public static void token_sentiment(String text)
    {
    	CoreDocument document = token_pipeline.processToCoreDocument(text);
    	for (CoreLabel tok : document.tokens()) {
    		String token = tok.word();
    		if (token.matches("^[a-zA-Z0-9]+")) {
            	SentimentAnalyzer.run_sentiment_analysis(token);
    		}
        }
    }

    
    public static void main(String[] args) 
    {
    	String sample = "What a fascinating book! I really loved reading it. Thank you for recommending it to me!";
    	SentimentAnalyzer.init();
    	SentimentAnalyzer.run_sentiment_analysis(sample);
    	SentimentAnalyzer.token_sentiment(sample);
    }
}
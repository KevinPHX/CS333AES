package preprocessing;

import java.util.*;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.File;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.*; 

public class Pipeline {
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
	
	public static ArrayList<Tree> run_parser(Annotation annotation)
    {
		ArrayList<Tree> trees = new ArrayList<Tree>();
    	for(CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class))
	    {
			Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
			trees.add(tree);
	    } 
    	return trees;
    }
	
    public static List<List>[] run_sentiment_analysis(Annotation annotation, Boolean getDetails)
    {
    	
	    String category; 
	    int sentIdx = 0;
	    
	    ArrayList<List> score_info_verbose = new ArrayList<List>();
	    ArrayList<List> score_info = new ArrayList<List>();

	    
	    for(CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class))
	    {
			Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
			category = sentence.get(SentimentCoreAnnotations.SentimentClass.class);			
			if(getDetails == true){
				SimpleMatrix fivescores = RNNCoreAnnotations.getPredictions(tree);
				double veryNegative = fivescores.get(0);
				double negative = fivescores.get(1);
				double neutral = fivescores.get(2);
				double positive = fivescores.get(3);
				double veryPositive = fivescores.get(4);
				Octet<Integer,String,Double,Double,Double,Double,Double,String> info = 
						new Octet<Integer,String,Double,Double,Double,Double,Double,String>
						(sentIdx,category,veryNegative,negative,neutral,positive,veryPositive,"SENTENCE:\t\t"+sentence.toString());
				score_info_verbose.add(info.toList());
			} 
			else {
				Pair<Integer,String> info = new Pair<Integer,String>(sentIdx,category);
				score_info.add(info.toList());
			}
			sentIdx++;
	    }
	    List[] scores = new List[2];
	    scores[0] = score_info;
	    scores[1] = score_info_verbose;
	    return scores;

     }

    
    public static void token_sentiment(Annotation annotation)
    {
    	for (CoreLabel tok : annotation.get(CoreAnnotations.TokensAnnotation.class)) {
    		System.out.println(tok.word() + "\t" + tok.lemma() + "\t" + tok.tag());
    		String token = tok.word();
    		if (token.matches("^[a-zA-Z0-9]+")) {
    			Annotation token_annotation = Pipeline.annotate(token);
            	Pipeline.run_sentiment_analysis(token_annotation,false);
    		}
        }
    }

    public static void run_pipeline(String text, String filename){
        File token_file = new File("src/main/resources/token_level/"+filename);
        File parse_file = new File("src/main/resources/parse_trees/"+filename);
		File sent_file = new File("src/main/resources/sentence_sentiment/"+filename);

		Annotation annotation = Pipeline.annotate(text);
		
		// Annotate each token with its covering sentence 
		List<Integer> tokenSentIdx = new ArrayList<Integer>(); 
		int sentIdx = 0; 
		for(CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class))
	    {
			Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
			List tokens = tree.getLeaves();
			for (Object token : tokens) {
				tokenSentIdx.add(sentIdx);
			}
			sentIdx++;
	    }
		// Find the lemma, POS tag, and sentiment for each token 
		// outputs as [index of covering sentence, token, lemma, POS, sentiment]
		PrintWriter token_output = null;
		try {
			token_output = new PrintWriter(token_file);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		int tokenIdx = 0; 
		for (CoreLabel tok : annotation.get(CoreAnnotations.TokensAnnotation.class)) {
    		String token = tok.word();
			int covering_sent = tokenSentIdx.get(tokenIdx);
    		if (token.matches("^[a-zA-Z0-9]+")) {
    			Annotation token_annotation = Pipeline.annotate(token);
            	List<List>[] sent_analysis = Pipeline.run_sentiment_analysis(token_annotation,false);
            	List<List> score_info = sent_analysis[0]; 
            	int idx = 0;
            	for(List sent_info : score_info) {
            		for(Object item : sent_info) {
            			if(idx == 1) {
            				String sentiment = item.toString();
            				Sextet<Integer,Integer,String,String,String,String> info = 
            						new Sextet<Integer,Integer,String,String,String,String>
            						(covering_sent,tok.index(),token, tok.lemma(), tok.tag(), sentiment);
            				token_output.println(info.toList());
            				
            			}
                		idx++;
            		}
            	}
    		}
    		else {
    			Sextet<Integer,Integer,String,String,String,String> info = 
						new Sextet<Integer,Integer,String,String,String,String>
						(covering_sent,tok.index(),token, tok.lemma(), tok.tag(), "");
				token_output.println(info.toList());    		
				}
    		tokenIdx++; 
        }
    	token_output.close();
    	System.out.println("Finished processing tokens of " + filename);
		
		// Sentiment scores and parse tree of each sentence 

        PrintWriter sent_output = null;
		try {
			sent_output = new PrintWriter(sent_file);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		PrintWriter parse_output = null;
		try {
			parse_output = new PrintWriter(parse_file);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}


		List<List>[] sent_analysis = Pipeline.run_sentiment_analysis(annotation,true);
    	List<List> score_info = sent_analysis[1];
    	ArrayList<Tree> parse_trees = Pipeline.run_parser(annotation);
    	sentIdx = 0;
    	for(List sent_info : score_info) {
    		sent_output.println(sent_info);
    		parse_output.println(parse_trees.get(sentIdx).toString());
        	sentIdx++;
    	}
    	sent_output.close();
    	parse_output.close();
    	System.out.println("Finished processing sentiments of " + filename);
    	System.out.println("Finished processing parse trees of " + filename);


    }
    
    public static void main(String[] args) throws IOException 
    {
    	Pipeline.init();
    	String dirname = "src/main/resources/essays";
    	File dir = new File(dirname);
    	File[] files = dir.listFiles();
	    for (File f_name : files) {
	    	String filename = f_name.toString();
	    	String essay_name = filename.replace(dirname + "/","");
	    	String essay = new String(Files.readAllBytes(Paths.get(filename)));
	    	Pipeline.run_pipeline(essay, essay_name);
	    	
	    }		
    	
    }
}
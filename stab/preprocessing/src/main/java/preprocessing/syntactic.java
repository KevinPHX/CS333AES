package preprocessing;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.File;
import java.io.PrintWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;


import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.CoreMap;

public class syntactic {
public static StanfordCoreNLP pipeline;
    
	public static void init() 
    {
        Properties props = new Properties();
		// default parser is PCFG 
        props.setProperty("annotators", "tokenize,ssplit,parse");
        pipeline = new StanfordCoreNLP(props);

    }

	public static Annotation annotate(String text){
		return pipeline.process(text);
	}
	
	
	public static Tree get_LCA(Tree tree, Tree u, Tree v) {
		// if u is root
		if(u == tree) {
			return u;
		}
		else if (v == tree) {
			return v;
		}
		
		Tree uAncestor = u.ancestor(1,tree);
		Tree vAncestor = v.ancestor(1,tree);
//		System.out.println(uAncestor);
//		System.out.println(vAncestor);
		
		if (u == vAncestor) { // u is the ancestor of v 
			return u;
		}
		else if (v == uAncestor) { // v is the ancestor of u 
			return v;
		}
		
		// case where u and v have an ancestor that is not u or v 
		// return as soon as we find a common ancestor 
		if (uAncestor == vAncestor) {
			return uAncestor;
		}
		return syntactic.get_LCA(tree, uAncestor, vAncestor);
	} 
	
	public static void LCA(String text, String filename)
    {
    	Annotation annotation = syntactic.annotate(text);

		File lca_file = new File("src/main/resources/LCA_info/"+filename);
		PrintWriter lca_output = null;
		try {
			lca_output = new PrintWriter(lca_file);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		int sentIdx = 0;
    	for(CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class))
	    {
			Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
			int depth = tree.depth();
			List<Tree> tokens = tree.getLeaves();
			for (int i=0; i < tokens.size(); i++){ // iterate through the tokens 
				Tree current = tokens.get(i);
				
				// set defaults 
				String preceding_label = "";
				String following_label = "";
				String c_preceding = "";
				String c_following = "";
				float lcaPath_preceding = -1;
				float lcaPath_following = -1; 

				
				String c_token = current.parent(tree).label().toString();
				if (i < tokens.size()-1) { // is not the last token 
					Tree following = tokens.get(i+1);
					following_label = following.toString();
					c_following = following.parent(tree).label().toString();
					Tree lca_following = syntactic.get_LCA(tree, current, following);
					lcaPath_following = (float) lca_following.depth(current) / depth;	
				}
				if (i > 0) { // is not the first token 
					Tree preceding = tokens.get(i-1);
					preceding_label = preceding.toString();
					c_preceding = preceding.parent(tree).label().toString();
					Tree lca_preceding = syntactic.get_LCA(tree, current, preceding);
					lcaPath_preceding = (float) lca_preceding.depth(current) / depth;
				}
				// index of covering sentence, current token, current token's label, constituent of current,
				// preceding token, constituent of preceding, lcaPath_preceding 
				// following token, constituent of following, lcaPath_following 
				String info = String.format("%d\t%s\t%s\t%s\t%s\t%s\t%f\t%s\t%s\t%f", 
						sentIdx, current.toString(), current.label().toString(), c_token,
						preceding_label, c_preceding, lcaPath_preceding,
						following_label, c_following, lcaPath_following);
				lca_output.println(info);
			}
			sentIdx ++; 
	    } 
    	lca_output.close();
    	System.out.println("Finished processing " + filename);
    }
	
	 public static void main(String[] args) throws IOException {
		 syntactic.init();
		 String dirname = "src/main/resources/essays";
	     File dir = new File(dirname);
	     File[] files = dir.listFiles();
	     for (File f_name : files) {
	    	String filename = f_name.toString();
	    	String essay_name = filename.replace(dirname + "/","");
	    	String essay = new String(Files.readAllBytes(Paths.get(filename)));
	    	syntactic.LCA(essay, essay_name);
		    }		
	 } 

}

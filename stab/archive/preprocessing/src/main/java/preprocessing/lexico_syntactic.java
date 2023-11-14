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

public class lexico_syntactic {
	public static HeadFinder head;
	public static HashMap<Integer,String> heads; // map node number in constituency tree to lexical head  
	public static PrintWriter lex_output;
	
	public static void init() 
    {
		syntactic.init(); 
		head = new UniversalSemanticHeadFinder();
		heads = new HashMap<Integer,String>();
    }
	
	
	public static String find_head(Tree node, Tree root){ // with memoization 
		int index = node.nodeNumber(root);

		if (node.isLeaf() == true) { // is a token, so head is itself  
			heads.put(index, node.label().toString());
			return node.label().toString(); 
		}
		else {
			Tree head_daughter = head.determineHead(node);
			int daughter_idx = head_daughter.nodeNumber(root);
			String daughter_val = heads.get(daughter_idx);
			if (daughter_val != null) {
				heads.put(index, daughter_val);
				return daughter_val;
			}
			else {
				daughter_val = find_head(head_daughter,root);
				heads.put(index, daughter_val);
				return daughter_val;
				
			}
		}
	} 
	
	public static Tree uppermost(Tree node, Tree root) {
		Tree parent = node.ancestor(1,root); 
		String parent_head = heads.get(parent.nodeNumber(root));
		String node_head = heads.get(node.nodeNumber(root));
		if (parent == root) { 
			// edge case where the uppermost ancestor with the same lexical head is the root 
			return root;
		}
		if (parent_head != node_head) { 
			// the parent of the current node has a different lexical head, 
			// so the uppermost ancestor with the same head as the current node is itself. 
			return node;
		}
		else { // continue searching up the tree 
			return uppermost(parent,root);
		}
		
	}
	
	public static void run_parser(Annotation annotation, String filename) {
		File lex_file = new File("src/main/resources/syntactic/lexico_syntactic/"+filename);
		lex_output = null;
		try {
			lex_output = new PrintWriter(lex_file);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		for(CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class))
	    {
			heads = new HashMap<Integer, String>(); 
			Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);

			// find the lexical head of each node in the tree 
			List<Tree> nodes = tree.postOrderNodeList();
			List<String> node_labels = new ArrayList<String>();
			for (Tree node : nodes) {
				node_labels.add(String.format("%s\t%s", node.label().toString(), node.nodeNumber(tree)));
				String lex_head = lexico_syntactic.find_head(node, tree);
			}
			
			HashMap<Tree,Tree> uppermost_nodes = new HashMap<Tree,Tree>();
			for (Tree token : tree.getLeaves()) { 
				// get the uppermost node n with lexical head t for each token t 
				Tree uppermost = uppermost(token,tree);
				uppermost_nodes.put(token,uppermost);
				// use extra newline to separate out the info for each token 
			}
			
			for (Tree token : tree.getLeaves()) { 
				// write output and get children of uppermost node n and their lexical heads 
				Tree n = uppermost_nodes.get(token);
				Tree child;
				int child_idx = Integer.MAX_VALUE; 
				
				lex_output.println(token.label() + "\t" + n.label());
				for (Tree c: n.getChildrenAsList()) {
					String label = c.label().toString();
					if (label.equals(token.label().toString())) {
						continue;
					}
					int c_idx = c.nodeNumber(tree);
					String lex_head = heads.get(c_idx);
					if (lex_head.equals(token.label().toString())) {
						child_idx = c_idx; 
						lex_output.println(token.label() + "\t" + "child_of_uppermost" + "\t" + label);
					}
					if (c_idx > child_idx) {
						// right sibling 
						lex_output.println(token.label() + "\t"+ "right_sibling_of_child " + "\t" + label + "\t" + lex_head);
						break;
					} 
				}
									
			}
			lex_output.println(node_labels);
			lex_output.println(heads);
			lex_output.print("\n");
		}
		lex_output.close();
		System.out.println("Finished processing " + filename);
		
	}

}


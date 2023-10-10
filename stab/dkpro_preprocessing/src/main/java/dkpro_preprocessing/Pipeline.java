package dkpro_preprocessing;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static org.apache.uima.fit.factory.CollectionReaderFactory.createReaderDescription;
import static org.apache.uima.fit.pipeline.SimplePipeline.runPipeline;

import de.tudarmstadt.ukp.dkpro.core.io.conll.Conll2006Writer;
import de.tudarmstadt.ukp.dkpro.core.io.text.TextReader;
import de.tudarmstadt.ukp.dkpro.core.matetools.MateLemmatizer;
import de.tudarmstadt.ukp.dkpro.core.languagetool.LanguageToolSegmenter;
import de.tudarmstadt.ukp.dkpro.core.stanfordnlp.StanfordPosTagger;
import de.tudarmstadt.ukp.dkpro.core.stanfordnlp.StanfordParser;

public class Pipeline {

  public static void main(String[] args) throws Exception {
    runPipeline(
        createReaderDescription(TextReader.class,
            TextReader.PARAM_SOURCE_LOCATION, "src/main/resources/essay01.txt",
            TextReader.PARAM_LANGUAGE, "en"),
        createEngineDescription(LanguageToolSegmenter.class),
        createEngineDescription(StanfordPosTagger.class),
        createEngineDescription(MateLemmatizer.class),
        createEngineDescription(StanfordParser.class),
        createEngineDescription(Conll2006Writer.class,
            Conll2006Writer.PARAM_TARGET_LOCATION, "."));
  }
}
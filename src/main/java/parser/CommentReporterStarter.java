package tutorial;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.comments.Comment;

public class CommentReporterStarter {
    private static final String FILE_PATH = "src/main/resources/sample/ReversePolishNotation.java";

    public static void main(String[] args) throws FileNotFoundException {
        CompilationUnit cu = StaticJavaParser.parse(new File(FILE_PATH));
        List<Comment> comments = cu.getAllContainedComments();
        System.out.print(comments);
    }
}
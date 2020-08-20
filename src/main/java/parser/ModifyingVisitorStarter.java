package tutorial;

import java.io.File;
import java.io.FileNotFoundException;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;

public class ModifyingVisitorStarter {
    private static final String FILE_PATH = "src/main/resources/sample/ReversePolishNotation.java";

    public static void main(String[] args) throws FileNotFoundException {
        System.out.println("Hello world");

        CompilationUnit cu = StaticJavaParser.parse(new File(FILE_PATH));

        System.out.println(cu.toString());
    }

}
package tutorial;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.printer.lexicalpreservation.LexicalPreservingPrinter;

class LexicalPreservingStarter {
public static void main(String[] args) {
    String code = "// Hey, this is a comment\n\n\n// Another one\n\nclass A { }";
    CompilationUnit cu = StaticJavaParser.parse(code);
    LexicalPreservingPrinter.setup(cu);
    
    System.out.println(LexicalPreservingPrinter.print(cu));
}
}
package tutorial;

import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.comments.JavadocComment;
import com.github.javaparser.printer.PrettyPrintVisitor;
import com.github.javaparser.printer.PrettyPrinter;
import com.github.javaparser.printer.PrettyPrinterConfiguration;
import com.github.javaparser.ast.expr.MarkerAnnotationExpr;
import com.github.javaparser.ast.expr.SingleMemberAnnotationExpr;
import com.github.javaparser.ast.expr.NormalAnnotationExpr;

class PrettyPrintComplete {
    public static void main(String[] args) {
        ClassOrInterfaceDeclaration myClass = new ClassOrInterfaceDeclaration();
        myClass.setComment(new JavadocComment("A very cool class!"));
        myClass.setName("MyClass");
        myClass.addField("String", "foo");
        myClass.addAnnotation("MySecretAnnotation");

        PrettyPrinterConfiguration conf = new PrettyPrinterConfiguration();
        conf.setIndentSize(4);
        conf.setPrintComments(false);
        conf.setVisitorFactory(prettyPrinterConfiguration -> new PrettyPrintVisitor(conf) {
            @Override
            public void visit(MarkerAnnotationExpr n, Void arg) {
                // ignoreW
            }

            @Override
            public void visit(SingleMemberAnnotationExpr n, Void arg) {
                // ignore
            }

            @Override
            public void visit(NormalAnnotationExpr n, Void arg) {
                // ignore
            }
        });

        PrettyPrinter prettyPrinter = new PrettyPrinter(conf);
        System.out.println(prettyPrinter.print(myClass));
    }
}
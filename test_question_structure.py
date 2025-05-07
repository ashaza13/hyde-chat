#!/usr/bin/env python3
"""
Test script for the Question data structure and CSV handling.
"""

from question_structure import QuestionTree, AnswerType

def main():
    # Load questions from CSV
    csv_path = "sample_questions.csv"
    print(f"Loading questions from {csv_path}...")
    tree = QuestionTree.from_csv(csv_path)
    
    # Display basic statistics
    print(f"Loaded {len(tree.questions)} questions")
    print(f"Root questions: {tree.root_questions}")
    
    # Display all questions with their parent relationships and dependencies
    print("\nQuestion Hierarchy:")
    for question in sorted(tree.get_all_questions(), key=lambda q: q.id):
        parent_info = f" (parent: {question.parent_id})" if question.parent_id else ""
        dep_info = f" [dependencies: {', '.join(question.dependencies)}]" if question.dependencies else ""
        print(f"  {question.id}{parent_info}{dep_info}: {question.text}")
    
    # Show dependencies for a specific question
    question_id = "1"
    print(f"\nDependencies for Question {question_id}:")
    q1 = tree.get_question(question_id)
    if q1:
        print(f"  Question: {q1.text}")
        for dep_id in q1.dependencies:
            dep = tree.get_question(dep_id)
            print(f"  - Depends on {dep_id}: {dep.text}")
    
    # Show children for a specific question
    print(f"\nChildren of Question {question_id}:")
    children = tree.get_children(question_id)
    for child in children:
        print(f"  - {child.id}: {child.text}")
    
    # Show the order in which questions should be processed
    print("\nQuestions in dependency order:")
    for i, question in enumerate(tree.get_questions_in_order(), 1):
        print(f"  {i}. {question.id}: {question.text}")
    
    # Demonstrate updating answers
    print("\nUpdating answers for question 1a...")
    q1a = tree.get_question("1a")
    if q1a:
        q1a.rag_answer = AnswerType.YES
        q1a.rag_confidence = 0.95
        q1a.rag_explanation = "The financial statements follow GASB 34 requirements for basic financial reporting."
        
        q1a.context_answer = AnswerType.YES
        q1a.context_confidence = 0.90
        q1a.context_explanation = "The statements include all required sections defined by GASB 34."
        
        print(f"  RAG Answer: {q1a.rag_answer}, Confidence: {q1a.rag_confidence}")
        print(f"  Context Answer: {q1a.context_answer}, Confidence: {q1a.context_confidence}")
    
    # Save the updated tree back to CSV
    output_csv = "updated_questions.csv"
    tree.to_csv(output_csv)
    print(f"\nSaved updated questions to {output_csv}")
    
    # Show dependency graph
    print("\nDependency graph:")
    dep_graph = tree.get_dependency_graph()
    for qid, dependent_questions in sorted(dep_graph.items()):
        if dependent_questions:
            print(f"  {qid} is required by: {', '.join(sorted(dependent_questions))}")


if __name__ == "__main__":
    main() 
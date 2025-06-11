from typing import Dict, List, Optional, Set
from enum import Enum
import re
import csv
import pandas as pd


class AnswerType(str, Enum):
    YES = "Yes"
    NO = "No"
    NA = "N/A"
    UNKNOWN = "Unknown"


class Question:
    """
    Represents an audit question with potential dependencies on other questions.
    """
    def __init__(
        self, 
        id: str, 
        text: str, 
        truth: Optional[AnswerType] = None,
        parent_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ):
        """
        Initialize a Question object.
        
        Args:
            id: Unique identifier for the question (e.g., "1", "1a", "2b")
            text: The question text
            truth: The known correct answer (if available)
            parent_id: ID of the parent question (e.g., parent of "1a" is "1")
            dependencies: List of question IDs that must be answered before this one
        """
        self.id = id
        self.text = text
        self.truth = truth if truth else AnswerType.UNKNOWN
        self.parent_id = parent_id
        self.dependencies = dependencies if dependencies else []
        
        # Answers from different approaches
        self.rag_answer: Optional[AnswerType] = None
        self.rag_explanation: Optional[str] = None
        self.rag_confidence: Optional[float] = None
        
        self.hyde_answer: Optional[AnswerType] = None
        self.hyde_explanation: Optional[str] = None
        self.hyde_confidence: Optional[float] = None
        
        self.context_answer: Optional[AnswerType] = None
        self.context_explanation: Optional[str] = None
        self.context_confidence: Optional[float] = None
        
        self.tree_traverse_answer: Optional[AnswerType] = None
        self.tree_traverse_explanation: Optional[str] = None
        self.tree_traverse_confidence: Optional[float] = None
    
    def __str__(self) -> str:
        return f"Question {self.id}: {self.text}"


class QuestionTree:
    """
    Manages a collection of questions with their dependencies.
    """
    def __init__(self):
        self.questions: Dict[str, Question] = {}
        self.root_questions: List[str] = []  # Question IDs that have no parent
    
    def add_question(self, question: Question) -> None:
        """Add a question to the tree."""
        self.questions[question.id] = question
        
        # Update root questions list
        if not question.parent_id:
            self.root_questions.append(question.id)
    
    def get_question(self, id: str) -> Optional[Question]:
        """Get a question by ID."""
        return self.questions.get(id)
    
    def get_dependencies(self, id: str) -> List[Question]:
        """Get all dependency questions for a given question ID."""
        question = self.get_question(id)
        if not question:
            return []
        
        return [self.questions[dep_id] for dep_id in question.dependencies if dep_id in self.questions]
    
    def get_children(self, id: str) -> List[Question]:
        """Get all direct child questions."""
        return [q for q in self.questions.values() if q.parent_id == id]
    
    def get_all_questions(self) -> List[Question]:
        """Get all questions in the tree."""
        return list(self.questions.values())
    
    def get_questions_in_order(self) -> List[Question]:
        """
        Return questions in an order where dependencies come before the questions that depend on them.
        """
        result = []
        visited = set()
        
        def visit(question_id):
            if question_id in visited:
                return
            
            question = self.questions[question_id]
            
            # Visit dependencies first
            for dep_id in question.dependencies:
                if dep_id in self.questions and dep_id not in visited:
                    visit(dep_id)
            
            visited.add(question_id)
            result.append(question)
        
        # Visit from each root question
        for question_id in self.questions:
            if question_id not in visited:
                visit(question_id)
        
        return result
    
    @classmethod
    def from_csv(cls, csv_path: str) -> 'QuestionTree':
        """
        Create a QuestionTree from a CSV file.
        
        Expected CSV format:
        id,question,truth,rag,context,hyde,tree_traverse
        """
        tree = cls()
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Process questions to identify hierarchy and dependencies
        parent_map = {}
        dependencies_map = {}
        
        # First pass - identify parent relationships
        for _, row in df.iterrows():
            question_id = str(row['id']).strip()
            
            # Identify parent relationship from ID (e.g., "1a" has parent "1")
            parent_id = None
            if re.match(r'^\d+[a-z]+$', question_id):  # IDs like "1a", "2b", etc.
                base_id = re.match(r'^\d+', question_id).group(0)
                if base_id != question_id:
                    parent_id = base_id
                    
                    # Child questions are dependencies of their parent
                    if parent_id not in dependencies_map:
                        dependencies_map[parent_id] = []
                    dependencies_map[parent_id].append(question_id)
            
            parent_map[question_id] = parent_id
        
        # Second pass - create Question objects
        for _, row in df.iterrows():
            question_id = str(row['id']).strip()
            
            # Convert truth value to AnswerType if it exists
            truth = None
            if 'truth' in row and pd.notna(row['truth']) and str(row['truth']).strip() != '':
                truth_str = str(row['truth']).strip()
                if truth_str.upper() == "YES":
                    truth = AnswerType.YES
                elif truth_str.upper() == "NO":
                    truth = AnswerType.NO
                elif truth_str == "N/A" or truth_str.upper() == "N/A":
                    truth = AnswerType.NA
                elif truth_str.upper() == "UNKNOWN":
                    truth = AnswerType.UNKNOWN
            
            question = Question(
                id=question_id,
                text=row['question'],
                truth=truth,
                parent_id=parent_map.get(question_id),
                dependencies=dependencies_map.get(question_id, [])
            )
            
            tree.add_question(question)
        
        return tree
    
    def to_csv(self, csv_path: str) -> None:
        """
        Write the question tree to a CSV file.
        """
        rows = []
        for question in self.questions.values():
            # Handle truth value output properly - include all valid truth values
            truth_value = ''
            if question.truth and question.truth != AnswerType.UNKNOWN:
                truth_value = question.truth.value
            
            row = {
                'id': question.id,
                'question': question.text,
                'truth': truth_value,
                'rag': question.rag_answer.value if question.rag_answer else '',
                'rag_confidence': question.rag_confidence if question.rag_confidence is not None else '',
                'rag_explanation': question.rag_explanation if question.rag_explanation else '',
                'context': question.context_answer.value if question.context_answer else '',
                'context_confidence': question.context_confidence if question.context_confidence is not None else '',
                'context_explanation': question.context_explanation if question.context_explanation else '',
                'hyde': question.hyde_answer.value if question.hyde_answer else '',
                'hyde_confidence': question.hyde_confidence if question.hyde_confidence is not None else '',
                'hyde_explanation': question.hyde_explanation if question.hyde_explanation else '',
                'tree_traverse': question.tree_traverse_answer.value if question.tree_traverse_answer else ''
            }
            rows.append(row)
        
        # Sort rows by ID for better readability
        rows.sort(key=lambda x: x['id'])
        
        # Define all possible fieldnames
        fieldnames = [
            'id', 'question', 'truth', 
            'rag', 'rag_confidence', 'rag_explanation',
            'context', 'context_confidence', 'context_explanation', 
            'hyde', 'hyde_confidence', 'hyde_explanation',
            'tree_traverse'
        ]
        
        # Write to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Return a dictionary representation of the dependency graph.
        Keys are question IDs, values are sets of questions that depend on this question.
        """
        graph = {qid: set() for qid in self.questions}
        
        for qid, question in self.questions.items():
            for dep_id in question.dependencies:
                if dep_id in graph:
                    graph[dep_id].add(qid)
        
        return graph 
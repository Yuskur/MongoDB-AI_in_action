import {React, useState, useEffect} from "react";
import './Explore.css';
import Table from 'react-bootstrap/Table';
import { motion, AnimatePresence } from 'framer-motion';
import Spinner from 'react-bootstrap/Spinner';

const Explore = () => {
    const columnGroups = [
        ['Age', 'Course', 'Gender', 'CGPA', 'Stress_Level', 'Depression_Score'],
        ['Anxiety_Score', 'Sleep_Quality', 'Physical_Activity', 'Diet_Quality', 'Social_Support'],
        ['Relationship_Status', 'Substance_Use', 'Counseling_Service_Use', 'Family_History', 'Chronic_Illness'],
        ['Financial_Stress', 'Extracurricular_Involvement', 'Semester_Credit_Load', 'Residence_Type'],
    ];

    const [currentGroup, setCurrentGroup] = useState(0);
    const columnsToShow = columnGroups[currentGroup];
    const [direction, setDirection] = useState(-1);

    const [isLoading, setLoading] = useState(false);
    
    // Query taken from the user
    const [query, setQuery] = useState('');
    
    // Storing data
    const [data, setData] = useState([]);

    useEffect(() => {
        const savedQueryData = localStorage.getItem("saved-query-data");
        if(savedQueryData){
            setData(JSON.parse(savedQueryData));
        }
    }, [])
    
    const getQueryData = () => {
        setLoading(true);
        fetch('http://127.0.0.1:5000/api/query', {
            method: 'POST',
            headers: {
                'Content-type': 'application/json'
            },
            body: JSON.stringify({query}),
            credentials: 'include'
        }).then(response => {
            if(response.ok){
                setQuery('');
                return response.json();
            }
        }).then(data => {
            const queryData = data.map(entry => JSON.parse(entry.record));
            setData(queryData);
            localStorage.setItem("saved-query-data", JSON.stringify(queryData));
            setLoading(false);
        }).catch(error => console.error(error))
    }

    function QueryTable(){
        return(
            <div>
                <div className="d-flex justify-content-between mb-2">
                    <button
                        className="page-button"
                        variant="none"
                        onClick={() => {
                                setCurrentGroup(prev => Math.max(prev - 1, 0))
                                setDirection(-1);
                            }
                        }
                        disabled={currentGroup === 0}
                    >
                    Prev
                    </button>
                    <span>Page {currentGroup + 1} of {columnGroups.length}</span>
                    <button
                        className="page-button"
                        variant="none"
                        onClick={() => {
                                setCurrentGroup(prev => Math.min(prev + 1, columnGroups.length - 1))
                                setDirection(1);
                            }
                        }
                        disabled={currentGroup === columnGroups.length - 1}
                    >
                    Next 
                    </button>
                </div>

                <AnimatePresence mode="wait">
                    <motion.div
                        key={currentGroup}
                        initial={{ x: direction * 100, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: -direction * 100, opacity: 0 }}
                        transition={{ duration: 0.05 }}
                    >
                        <Table striped bordered hover responsive>
                            <thead>
                                <tr>
                                    {columnsToShow.map(col => <th key={col}>{col}</th>)}
                                </tr>
                            </thead>
                            <tbody>
                                {data.map((record, index) => (
                                <tr className="record" key={index}>
                                    {columnsToShow.map(col => <td key={col}>{record[col]}</td>)}
                                </tr>
                                ))}
                            </tbody>
                        </Table>
                    </motion.div>
                </AnimatePresence>
            </div>
        );
    }

    function Loading(){
        return(
            <Spinner animation="border" role="status">
                <span className="visually-hidden">Loading...</span>
            </Spinner>
        );
    }
    
    return(
        <div className="explore-container">
            <div className="content-container">
                {isLoading ? <Loading /> : <QueryTable/>}
            </div>
            <div className="query-bar">
                <textarea 
                    placeholder="Type your query..." 
                    className="query-input"
                    value={query}
                    onChange={(change) => setQuery(change.target.value)}
                />
                <button 
                className="page-button" 
                id="submit"
                onClick={() => getQueryData()}
                >submit</button>
            </div>
        </div>
    )
}

export default Explore;
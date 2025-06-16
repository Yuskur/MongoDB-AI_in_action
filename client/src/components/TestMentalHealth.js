import React, { useState, useEffect } from "react";
import "./TestMentalHealth.css";
import CloseButton from 'react-bootstrap/CloseButton';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';

function TestMentalHealth(props) {


    const [age, setAge] = useState("");
    const [course, setCourse] = useState("");
    const [gender, setGender] = useState("");
    const [cgpa, setCgpa] = useState("");
    const [stress, setStress] = useState("");
    const [sleep, setSleep] = useState("");
    const [activity, setActivity] = useState("");
    const [diet, setDiet] = useState("");
    const [support, setSupport] = useState("");
    const [relationship, setRelationship] = useState("");
    const [substance, setSubstance] = useState("");
    const [counseling, setCounseling] = useState("");
    const [familyHistory, setFamilyHistory] = useState("");
    const [chronic_illness, setIllness] = useState("");
    const [financial, setFinancial] = useState("");
    const [extracurricular, setExtracurricular] = useState("");
    const [credits, setCredits] = useState("");
    const [residence, setResidence] = useState("");
    const [validated, setValidated] = useState(false);

    useEffect(() => {
        if (props.trigger) {
            setAge("");
            setCourse("");
            setGender("");
            setCgpa("");
            setStress("");
            setSleep("");
            setActivity("");
            setDiet("");
            setSupport("");
            setRelationship("");
            setSubstance("");
            setCounseling("");
            setFamilyHistory("");
            setIllness("");
            setFinancial("");
            setExtracurricular("");
            setCredits("");
            setResidence("");
            setValidated(false);
        }
    }, [props.trigger]);

    const handleTryClick = async() => {
        try{
            const response = await fetch('http://127.0.0.1:5000/api/predict-mental-health', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body:JSON.stringify({
                    age,
                    course,
                    gender,
                    cgpa,
                    stress,
                    sleep,
                    activity,
                    diet,
                    support,
                    relationship,
                    substance,
                    counseling,
                    familyHistory,
                    chronic_illness,
                    financial,
                    extracurricular,
                    credits,
                    residence
                }),
                credentials: 'include'
            });

            const result = await response.json();
            return result;
        } catch(error){
            alert("Error: " + error);
            return; 
        }
  }
    
    return props.trigger ? (
        <div className="test-mental-health-container">
            <Form
            className="test-mental-health-inner"
            autoComplete="off"
            noValidate
            validated={validated}
            onSubmit={async (e) => {
                const form = e.currentTarget;
                e.preventDefault();
                if (form.checkValidity() === false) {
                    e.preventDefault();
                    e.stopPropagation();
                } else {
                    const result = await handleTryClick();
                    console.log(result);
                    const prediction = result.prediction ? 'yes' : 'no';
                    alert(`Mental Health Risk Prediction: ${prediction}`);
                    props.setTrigger(false);
                    localStorage.removeItem("to-understand");
                    e.preventDefault();
                }
                setValidated(true);
            }}
            >
            <h3 className="form-title">Mental Health Test</h3>
            <CloseButton
                className="close-button"
                onClick={() => { 
                    props.setTrigger(false)
                    setValidated(false);
                    localStorage.removeItem("to-understand");
                }}
            />

            
            <Form.Group className="textBox" controlId="age">
                <Form.Label>Age:</Form.Label>
                <Form.Control type="number" required onChange={(e) => setAge(e.target.value)} />
            </Form.Group>

            <Form.Group className="textBox" controlId="course">
                <Form.Label>Course/Major:</Form.Label>
                <Form.Select required onChange={(e) => setCourse(e.target.value)}>
                <option value="">Select your course</option>
                <option value="Engineering">Engineering</option>
                <option value="Business">Business</option>
                <option value="Computer Science">Computer Science</option>
                <option value="Medical">Medical</option>
                <option value="Law">Law</option>
                <option value="Others">Others</option>
                </Form.Select>
            </Form.Group>

            <Form.Group className="textBox" controlId="gender">
                <Form.Label>Gender:</Form.Label>
                <Form.Select required onChange={(e) => setGender(e.target.value)}>
                <option value="">Select your gender</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Others">Others</option>
                </Form.Select>
            </Form.Group>

            <Form.Group className="textBox" controlId="cgpa">
                <Form.Label>CGPA:</Form.Label>
                <Form.Control
                type="number"
                step="0.01"
                required
                onChange={(e) => setCgpa(e.target.value)}
                />
            </Form.Group>

            <Form.Group className="textBox" controlId="sleepQuality">
                <Form.Label>Sleep Quality:</Form.Label>
                <Form.Select required onChange={(e) => setSleep(e.target.value)}>
                <option value="">Select</option>
                <option value="Poor">Poor</option>
                <option value="Average">Average</option>
                <option value="Good">Good</option>
                </Form.Select>
            </Form.Group>

            <Form.Group className="textBox" controlId="physicalActivity">
                <Form.Label>Physical Activity:</Form.Label>
                <Form.Select required onChange={(e) => setActivity(e.target.value)}>
                <option value="">Select</option>
                <option value="Low">Low</option>
                <option value="Moderate">Moderate</option>
                </Form.Select>
            </Form.Group>

            <Form.Group className="textBox" controlId="dietQuality">
                <Form.Label>Diet Quality:</Form.Label>
                <Form.Select required onChange={(e) => setDiet(e.target.value)}>
                <option value="">Select</option>
                <option value="Poor">Poor</option>
                <option value="Average">Average</option>
                <option value="Good">Good</option>
                </Form.Select>
            </Form.Group>

            <Form.Group className="textBox" controlId="socialSupport">
                <Form.Label>Social Support:</Form.Label>
                <Form.Select required onChange={(e) => setSupport(e.target.value)}>
                <option value="">Select</option>
                <option value="Low">Low</option>
                <option value="Moderate">Moderate</option>
                <option value="High">High</option>
                </Form.Select>
            </Form.Group>

            <Form.Group className="textBox" controlId="relationshipStatus">
                <Form.Label>Relationship Status:</Form.Label>
                <Form.Select required onChange={(e) => setRelationship(e.target.value)}>
                <option value="">Select</option>
                <option value="Single">Single</option>
                <option value="In a Relationship">In a Relationship</option>
                <option value="Married">Married</option>
                </Form.Select>
            </Form.Group>

            <Form.Group className="textBox" controlId="substanceUse">
                <Form.Label>Substance Use:</Form.Label>
                <Form.Select required onChange={(e) => setSubstance(e.target.value)}>
                <option value="">Select</option>
                <option value="Never">Never</option>
                <option value="Occasionally">Occasionally</option>
                <option value="Frequently">Frequently</option>
                </Form.Select>
            </Form.Group>

            <Form.Group className="textBox" controlId="counseling">
                <Form.Label>Counseling Service Use:</Form.Label>
                <Form.Select required onChange={(e) => setCounseling(e.target.value)}>
                <option value="">Select</option>
                <option value="Never">Never</option>
                <option value="Occasionally">Occasionally</option>
                <option value="Frequently">Frequently</option>
                </Form.Select>
            </Form.Group>

            <Form.Group className="textBox" controlId="familyHistory">
                <Form.Label>Family History of Mental Illness:</Form.Label>
                <Form.Select required onChange={(e) => setFamilyHistory(e.target.value)}>
                <option value="">Select</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                </Form.Select>
            </Form.Group>

            <Form.Group className="textBox" controlId="chronicIllness">
                <Form.Label>Chronic Illness:</Form.Label>
                <Form.Select required onChange={(e) => setIllness(e.target.value)}>
                <option value="">Select</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                </Form.Select>
            </Form.Group>

            <Form.Group className="textBox" controlId="financialStress">
                <Form.Label>Financial Stress (0–5):</Form.Label>
                <Form.Control
                type="number"
                min={0}
                max={5}
                required
                onChange={(e) => setFinancial(e.target.value)}
                />
            </Form.Group>

            <Form.Group className="textBox" controlId="extracurricular">
                <Form.Label>Extracurricular Involvement:</Form.Label>
                <Form.Select required onChange={(e) => setExtracurricular(e.target.value)}>
                <option value="">Select</option>
                <option value="Low">Low</option>
                <option value="Moderate">Moderate</option>
                <option value="High">High</option>
                </Form.Select>
            </Form.Group>

            <Form.Group className="textBox" controlId="creditLoad">
                <Form.Label>Semester Credit Load:</Form.Label>
                <Form.Control type="number" required onChange={(e) => setCredits(e.target.value)} />
            </Form.Group>

            <Form.Group className="textBox" controlId="residenceType">
                <Form.Label>Residence Type:</Form.Label>
                <Form.Select required onChange={(e) => setResidence(e.target.value)}>
                <option value="">Select</option>
                <option value="On-Campus">On-Campus</option>
                <option value="Off-Campus">Off-Campus</option>
                </Form.Select>
            </Form.Group>

            <button className="submit-button" variant="primary" type="submit">
                Test My Mental Health
            </button>
            </Form>
        </div>
    ) : "";
}

export default TestMentalHealth;
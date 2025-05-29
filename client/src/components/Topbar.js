import React from 'react';
import Nav from 'react-bootstrap/Nav';
import 'bootstrap/dist/css/bootstrap.min.css';
import './Topbar.css';
import { Link, useLocation } from 'react-router-dom';

const Topbar = () => {
    const location = useLocation();

    return (
        <Nav className="topbar-container" variant="pills" activeKey={location.pathname}>
            <Nav.Item>
                <Nav.Link as={Link} to='/' eventKey={'/'}>Explore</Nav.Link>
            </Nav.Item>
            <Nav.Item>
                <Nav.Link as={Link} to='/understand' eventKey={'/understand'}>Understand</Nav.Link>
            </Nav.Item>
        </Nav>
    );
}

export default Topbar;

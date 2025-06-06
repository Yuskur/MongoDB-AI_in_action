import React from "react";
import {Routes, Route } from "react-router-dom";
import Explore from "./pages/Explore";
import Understand from "./pages/Understand";

const Navigation = () => {
    return (
        <Routes>
            <Route path="/" element={<Explore />} />
            <Route path="/understand" element={<Understand />} />
        </Routes>
    );
};

export default Navigation;

// All the supporting functions for visualizing SciDoc and links
//


function mouseOverEvent(){
	$(this).attr("id");
}


function mouseOutEvent(){
	
}

function setupHandlers(){
	$("#citation").mouseenter(mouseOverEvent);
	$("#citation").mouseout(mouseOutEvent);
}
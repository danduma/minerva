
// All the supporting functions for visualizing SciDoc and links
//

var position_in_list=0;

function termOnClick(){
	$(".selected_term").removeClass("selected_term").addClass("highlight");
	$("."+$(this).attr("term_id")).removeClass("highlight").addClass("selected_term");
}

// Remove all highlights from all spans
function clearVisualization(){
	$(".highlight").removeClass("highlight").unbind("click");
	$(".selected_term").removeClass("selected_term").unbind("click");
	$(".selected_citation").removeClass("selected_citation");
}

// Add highlights
function showOverlaps(ref_id){ 
	// console.log("ref_id:"+ref_id);

	clearVisualization();
	$("."+ref_id).addClass("highlight").click(termOnClick);
}

//
function citationOnClick(){
	var ref_id=$(this).attr("ref_id");
	console.log(ref_id);
	$("#rightContent").html(token_data["ref_data"][ref_id]["full_html"]);
	$("#rightDocumentPanel").html(token_data["ref_data"][ref_id]["details"]);
	showOverlaps(ref_id);
	$(this).addClass("selected_citation");
}

function setupHandlers(){
	$(".in-collection_cit").click(citationOnClick);
	// $(".citation").mouseout(mouseOutEvent);
	// console.log("Handlers set up!");
}

function loadData(){
	// Check for the various File API support.
	if (window.File && window.FileReader && window.FileList && window.Blob) {
	  // Great success! All the File APIs are supported.
	} else {
	  alert('The File APIs are not fully supported in this browser.');
	}

	
	for (var i=0; i < file_data.length; i++){
	     $('#file_list')
	         .append($("<option></option>")
	         .attr("value",i)
	         .text(file_data[i]["details"])); 
	}

	loadFile(position_in_list);
}

function clearRightPanel(){
	$("#rightContent").html("");
	$("#rightDocumentPanel").html("");
}

function loadFile(file_num){
	if (file_num < file_data.length){
		clearRightPanel();

		$.getJSON(file_data[file_num]["json_file"], function(data){
			token_data=data;
			$("#leftContent").html(token_data["full_html"]);
			$("#leftDocumentPanel").html(file_data[file_num]["details"]);
			setupHandlers();
		})
	}
}

function nextFile(){
	console.log(position_in_list);
	console.log(file_data.length);
	if (position_in_list < file_data.length-1){
		position_in_list+=1;
		loadFile(position_in_list);
	}
}

function prevFile(){
	if (position_in_list > 0 && file_data.length > 0){
		position_in_list-=1;
		loadFile(position_in_list);
	}
}

function fileSelected(){
	var fl=$('#file_list');
	loadFile(fl.val());
}